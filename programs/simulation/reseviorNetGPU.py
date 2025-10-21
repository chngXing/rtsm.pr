import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import threading

class LIFNeurons(nn.Module):
    def __init__(self, dt=0.1, tau_m=10.0, v_th=-45.0, v_reset=-65.0, I_bias=-30.0, resting_period_e=2.0, resting_period_d2=1.0):
        super(LIFNeurons, self).__init__()
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.I_bias_value = I_bias
        self.resting_period_e = resting_period_e
        self.resting_period_d2 = resting_period_d2
        self.v = None
        self.spiked = None
        self.last_spike_time = None
        self.resting_period = None
        self.device = None
        self.num_neurons = None

    def reset_state(self, N, device, begin_v=0.0):
        self.v = torch.full((N, 1), begin_v, device=device)
        self.spiked = torch.zeros((N, 1), device=device)
        self.I_bias = torch.full((N, 1), self.I_bias_value, device=device)
        self.last_spike_time = torch.full((N, 1), -float('inf'), device=device)
        self.resting_period = torch.full((N, 1), self.resting_period_e, device=device) + torch.randn((N, 1), device=device) * self.resting_period_d2
        self.device = device
        self.num_neurons = N

    def resting(self, current_time):
        current_time_c = torch.full((self.num_neurons, 1), current_time, device=self.device)
        return (current_time_c - self.last_spike_time) < self.resting_period

    def reset_time(self):
        self.last_spike_time = torch.full((self.num_neurons, 1), -float('inf'), device=self.device)

    def forward(self, input_current, current_time):
        dv = (-self.v + input_current + self.I_bias) / self.tau_m * self.dt
        self.v += dv
        spiked = (self.v >= self.v_th).bool()
        resting_mask = self.resting(current_time).float()
        ones = torch.ones_like(resting_mask)
        spiked = spiked * (ones - resting_mask)
        reset_tensor = torch.full_like(self.v, self.v_reset)
        spike_times = torch.full((self.num_neurons, 1), current_time, device=self.device)
        self.v = torch.where(spiked.bool(), reset_tensor, self.v)
        self.last_spike_time = torch.where(spiked.bool(), spike_times, self.last_spike_time)
        self.spiked = spiked

        return spiked, self.v

class SpikingNeuronNetwork:
    def __init__(self, num_neurons,
                dt=0.1, tau_m=10.0,
                tau_r=2.0, tau_d=20.0,
                v_th=-50.0, v_reset=-65.0,
                I_bias=-55.0, p=25.0,
                q=60.0, p_w=0.1, lambda_=0.1,
                max_time_delta=200, delay=1.0,
                begin_v=None, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.v = v_th - v_reset
        self.I_bias = I_bias
        self.p = p
        self.q = q
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.time_turn = 0
        self.a = 2 * tau_r * tau_d / (tau_r + tau_d)
        alpha = 2.0
        mean_tau = 40.0 / self.dt
        scale = mean_tau / alpha
        delay = torch.distributions.Gamma(alpha, scale).sample((num_neurons, 1)) * delay
        self.max_time_delta = max_time_delta
        kernel = []
        for j in range(0, num_neurons):
            kernel.append([((mt.exp(-((-i + delay[j]) * self.dt / self.a) ** 2) / mt.sqrt(mt.pi)) / self.a * 2) for i in range(-max_time_delta - 1, 0)])
        self.time_delta = torch.tensor(kernel, device=self.device).T
        self.r = torch.zeros((num_neurons, 1), device=self.device)
        self.h = torch.zeros((num_neurons, 1), device=self.device)
        self.w = torch.randn(num_neurons, num_neurons, device=self.device) / (mt.sqrt(num_neurons) * p_w)
        self.eta = torch.rand((self.num_neurons, 1), device=self.device) * 2 - torch.full((self.num_neurons, 1), 1.0, device=self.device)
        self.phi = torch.zeros((self.num_neurons, 1), device=self.device)
        self.P = torch.eye(num_neurons, device=self.device) / lambda_
        self.neurons = LIFNeurons(dt=dt, tau_m=tau_m, v_th=v_th, v_reset=v_reset, I_bias=I_bias)
        self.neurons.reset_state(num_neurons, self.device, begin_v if begin_v is not None else self.v_reset)
        self.spike_history = []

    def reset(self, delay=1.0):
        self.time_turn = 0
        self.r = torch.zeros((self.num_neurons, 1), device=self.device)
        self.h = torch.zeros((self.num_neurons, 1), device=self.device)
        self.phi = torch.zeros((self.num_neurons, 1), device=self.device)
        self.P = torch.eye(self.num_neurons, device=self.device) / 0.1
        alpha = 5.0
        mean_tau = 20.0 / self.dt
        scale = mean_tau / alpha
        delay = torch.distributions.Gamma(alpha, scale).sample((self.num_neurons, 1)) * delay
        kernel = []
        for j in range(0, self.num_neurons):
            kernel.append([((mt.exp(-((-i + delay[j]) * self.dt / self.a) ** 2) / mt.sqrt(mt.pi)) / self.a * 2) for i in range(-self.max_time_delta - 1, 0)])
        self.time_delta = torch.tensor(kernel, device=self.device).T
        self.spike_history = []
        self.neurons.reset_state(self.num_neurons, self.device, self.v_reset)

        return self

    def delta_(self):
        # spike_history: list of tensors shape (N,1), len = time_turn
        begin_time = max(0, self.time_turn - self.max_time_delta)
        # stack -> (T, N, 1) -> squeeze -> (T, N) -> transpose -> (N, T)
        if self.time_turn == 0:
            return torch.zeros((self.num_neurons, 1), device=self.device)
        else:
            H = torch.stack(self.spike_history[begin_time:self.time_turn])  # (T, N, 1)
        if H.numel() == 0:
            return torch.zeros((self.num_neurons, 1), device=self.device)
        H = H.squeeze(-1).T   # (N, T)
        # time_delta should have shape (max_time_delta, N), we pick last T rows
        T = H.shape[1]
        td = self.time_delta[-T:, :]   # (T, N)
        # multiply per-time and sum over time -> result (N,)
        res = (H * td.T).sum(dim=1, keepdim=True)   # (N,1)
        return res

    def delta(self):
        begin_time = self.time_turn - 200 if self.time_turn - 200 > 0 else 0
        end_time = self.time_turn
        time_delta = self.time_delta[self.max_time_delta - (self.time_turn - begin_time): self.max_time_delta + 1]
        spike_history_tensor = torch.stack(self.spike_history[begin_time:end_time + 1]).squeeze(-1).T
        result = torch.sum(spike_history_tensor * time_delta.T, dim=1, keepdim=True)

        return result
    
    def step_r(self):
        self.h += self.dt * (-self.h / self.tau_r + self.delta().squeeze(0) / self.tau_r / self.tau_d)
        self.r += self.dt * (-self.r / self.tau_d + self.h)

    def step(self, input_current, noise_std=0.0):
        x = self.phi.T @ self.r
        if not torch.is_tensor(input_current):
            input_current = torch.full((self.num_neurons,), float(input_current), device=self.device)
        else:
            input_current = input_current.to(self.device)
        noise_std = noise_std if noise_std > 0 else 0.0
        noise = torch.randn((self.num_neurons, 1), device=self.device) * noise_std

        input_current = 25.0 * input_current.view(-1, 1)
        s = self.p * self.w @ self.r + self.q * self.eta * x + noise
        spike, v = self.neurons.forward(s + input_current, self.time_turn * self.dt)
        self.spike_history.append(spike.clone())
        self.step_r()
        self.time_turn += 1
        return x.item(), spike
    
    def train(self, target, learning_rate=1.0, train=True):
        r_vector = self.r
        Pr = self.P @ r_vector
        k = Pr / (1.0 + (r_vector.T @ Pr))
        error = target - (self.phi.T @ r_vector).squeeze(0)
        if self.time_turn % 50 == 0 and train:
            self.P -= k @ Pr.T
            Pr = self.P @ r_vector
            self.phi += learning_rate * error * Pr
        print(f'k:{k.sum().item():>10.5f} | error:{error.item():>10.5f}', end=' | ')

        return error

    def reset_time(self):
        self.time_turn = 0
        self.spike_history = []
        self.neurons.reset_time()

    def run(self, simtime, input=None, target=None, train=False, noise_std=60.0, noise_duration=40):
        max_steps = int(simtime / self.dt)
        spike_rec = torch.zeros(max_steps, self.num_neurons, 1, device=self.device)
        x_rec = torch.zeros(max_steps, device=self.device)
        errors = torch.zeros(max_steps, device=self.device)
        for t in range(max_steps):
            noise = noise_std if (t * self.dt) < noise_duration else 0.0
            x_rec[t], spike = self.step(input[(t % max_steps)], noise) if input is not None else self.step(0.0, noise)
            print(f'time: {self.time_turn * self.dt:.2f} ms', end='\r')
            spike_rec[t, :] = spike
            if target is not None and (t * self.dt) >= noise_duration:
                errors[t] = self.train(target[(t % max_steps)], train=train)

        return x_rec.cpu(), spike_rec.cpu(), errors.cpu()

def simple_validation(simtime=5000.0, num_neurons=2000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = 0.05
    t = np.arange(0, simtime, dt)
    A = [1. / 2., 1. / 10., 1. / 8., 1. / 14., 1. / 4., 1. / 20., 1. / 12., 1. / 18., 1. / 16., 1. / 6.]
    target = np.zeros_like(t)
    for i in range(len(A)):
        target += A[i] * np.sin(4 * mt.pi * (i + 1) * t / 1000.0)# + np.sin(2 * mt.pi * 1.2 * t / 200.0)# + np.sin(2 * mt.pi * 5 * t / 200.0) + np.sin(2 * mt.pi * 7 * t / 200.0) + np.sin(2 * mt.pi * 11 * t / 200.0) + np.sin(2 * mt.pi * 13 * t / 200.0) + np.sin(2 * mt.pi * 17 * t / 200.0)
    #input = np.abs(np.sin(2 * mt.pi * 1 * t / 200.0))
    net_without_delay = SpikingNeuronNetwork(dt=dt, num_neurons=num_neurons, device=device, delay=0.0)
    x_rec_train_nodelay, spikes_train_nodelay, errors_train_nodelay = net_without_delay.run(simtime * 0.8, input=None, train=True, target=target[:int(simtime/dt * 0.8)])
    x_rec_test_nodelay, spikes_test_nodelay, errors_test_nodelay = net_without_delay.run(simtime * 0.2, input=None, train=False, target=target[int(simtime/dt * 0.8):], noise_duration=0)
    x_rec_nodelay = (torch.cat((x_rec_train_nodelay, x_rec_test_nodelay), dim=0)).cpu().numpy()
    spikes_nodelay = (torch.cat((spikes_train_nodelay, spikes_test_nodelay), dim=0)).squeeze(-1).cpu().numpy()
    errors_nodelay = (torch.cat((torch.square(errors_train_nodelay), torch.square(errors_test_nodelay)), dim=0)).cpu().numpy()
    spike_counts_nodelay = spikes_nodelay.sum(axis=0)
    print(f"Device: {device} | spikes (no delay): {spike_counts_nodelay.mean():.2f}")
    spike_counts_nodelay_times = spikes_nodelay.sum(axis=1).reshape(-1, 2).sum(axis=1)

    net_with_delay = net_without_delay.reset(delay=1.0)
    x_rec_train, spikes_train, errors_train = net_with_delay.run(simtime * 0.8, input=None, train=True, target=target[:int(simtime/dt * 0.8)])
    x_rec_test, spikes_test, errors_test = net_with_delay.run(simtime * 0.2, input=None, train=False, target=target[int(simtime/dt * 0.8):], noise_duration=0)
    x_rec = (torch.cat((x_rec_train, x_rec_test), dim=0)).cpu().numpy()
    spikes = (torch.cat((spikes_train, spikes_test), dim=0)).squeeze(-1).cpu().numpy()
    errors = (torch.cat((torch.square(errors_train), torch.square(errors_test)), dim=0)).cpu().numpy()
    spike_counts = spikes.sum(axis=0)
    print(f"Device: {device} | spikes: {spike_counts.mean():.2f}")
    spike_counts_times = spikes.sum(axis=1).reshape(-1, 2).sum(axis=1)

    steps = np.arange(x_rec_train_nodelay.shape[0])
    plt.figure(figsize=(12, 9))
    plt.subplot(6, 1, 1)
    plt.plot(t, target, label='Target', color='blue')
    #plt.plot(t, input, label='Input', color='gray', alpha=0.5)
    plt.plot(t, x_rec_nodelay[-len(t):], label='Output (No Delay)', color='red', alpha=0.7)
    plt.plot(t, x_rec[-len(t):], label='Output (With Delay)', color='green', alpha=0.7)
    plt.title('Target vs Output')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.imshow(spikes_nodelay.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net_without_delay.num_neurons])
    plt.title('Spike Raster Plot (No Delay)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.bar(np.arange(len(spike_counts_nodelay_times)), spike_counts_nodelay_times, edgecolor='black')
    plt.title('Total Spike Counts Over Time (No Delay)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Spike Counts')
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.imshow(spikes.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net_with_delay.num_neurons])
    plt.title('Spike Raster Plot (With Delay)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.bar(np.arange(len(spike_counts_times)), spike_counts_times, edgecolor='black')
    plt.title('Total Spike Counts Over Time (With Delay)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Spike Counts')
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(t, errors_nodelay, label='MSE (No Delay)', color='red')
    plt.plot(t, errors, label='MSE (With Delay)', color='green')
    plt.title('Mean Squared Error Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('MSE')
    plt.show()


def test_lif(simtime=100.0, dt=0.1, num_neurons=10, noise_std=1000.0):
    steps = int(simtime / dt)
    neurons = LIFNeurons(dt=dt)
    neurons.reset_state(num_neurons, device='cpu', begin_v=-65.0)
    
    spike_record = torch.zeros((steps, num_neurons))
    v_record = torch.zeros((steps, num_neurons))
    
    for t in range(steps):
        # 每个神经元独立噪声输入
        input_current = torch.randn(num_neurons, 1) * noise_std
        spikes, v = neurons.forward(input_current=input_current, current_time=(t * dt))
        spike_record[t] = spikes.squeeze(-1)
        v_record[t] = v.squeeze(-1)
    
    t_array = np.arange(0, simtime, dt)
    
    # 画膜电位
    plt.figure(figsize=(12, 5))
    for n in range(num_neurons):
        plt.plot(t_array, v_record[:, n], label=f'Neuron {n+1}')
    plt.axhline(neurons.v_th, color='r', linestyle='--', label='v_th')
    plt.title("Membrane potentials")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.show()
    
    # 画 raster plot
    plt.figure(figsize=(12, 4))
    plt.imshow(spike_record.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, num_neurons])
    plt.title("Spike Raster Plot")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.show()

if __name__ == "__main__":
    #test_lif()
    simple_validation()
