import numpy as np
import matplotlib.pyplot as plt
import math as mt

class LIFneuron:
    def __init__(self, dt=0.1, tau_m=10.0, v_th=1.0, v_reset=0.0, I_bias=0.0, begin_v=None, resting_period=2.0):
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.v = begin_v if begin_v is not None else v_reset
        self.spiked = 0
        self.I_bias = I_bias
        self.last_spike_time = -np.inf
        self.resting_period = resting_period

    def reset(self):
        self.v = self.v_reset
        self.spiked = 0
    
    def resting(self, time):
        return (time - self.last_spike_time) < self.resting_period

    def step(self, input_current, current_time):
        dv = ( - self.v) * (self.dt / self.tau_m)  + input_current + self.I_bias
        self.v += dv
        if self.v >= self.v_th and not self.resting(current_time):
            self.spiked = 1
            self.v = self.v_reset
        else:
            self.spiked = 0
        return self.spiked

class spikingNeuronNetwork:
    def __init__(self, num_neurons,
                dt=0.1, tau_m=10.0,
                tau_r=2.0, tau_d=20.0,
                v_th=1.0, v_reset=0.0,
                I_bias=0.0, p=1.0,
                q=1.5, begin_v=None,
                p_w=0.1, lambda_=1.0):
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.I_bias = I_bias
        self.p = p
        self.q = q
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.time_turn = 0
        self.a = tau_r
        self.r = np.zeros(num_neurons)
        self.h = np.zeros(num_neurons)
        self.w = np.random.randn(num_neurons, num_neurons) * (mt.sqrt(num_neurons) / p_w)
        self.eta = np.random.rand(num_neurons)
        self.phi = np.zeros(num_neurons)
        self.P = np.eye(num_neurons) / lambda_
        self.neurons = [LIFneuron(dt,
                                tau_m,
                                v_th, 
                                v_reset, 
                                I_bias, 
                                begin_v) for _ in range(num_neurons)]
        self.v = v_th - v_reset
        self.spike_history = []
        self.time_history = []
    
    def derac(self, d):
        for i in range(0 if ((self.time_turn - 200) < 0) else int(self.time_turn - 200), int(self.time_turn)):
            if self.spike_history[i].size == 0:
                return None
            for j in range(self.num_neurons):
                d[j] += (mt.exp(-((self.time_turn - i) * self.dt / self.a) ** 2) / mt.sqrt(mt.pi)) * self.a * self.v if (self.spike_history[i][j] == 1) else 0

        return d

    def step_r(self):
        d = np.zeros(self.num_neurons)
        dh = -self.h / self.tau_r + self.derac(d) / self.tau_r
        self.h += dh * self.dt
        dr = -self.r / self.tau_d + self.h
        self.r += dr * self.dt
    
    def step(self, input_current=0.0):
        J = self.q * self.w @ self.r + self.p * self.eta * (self.phi @ self.r)
        s = J + input_current
        #s = J + 0.5 * np.random.randn(self.num_neurons)
        spike = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            spike[i] = self.neurons[i].step(s[i], self.time_turn * self.dt)

        self.spike_history.append(spike.copy())
        x = self.phi @ self.r
        self.step_r()
        self.time_turn += 1

        return x, spike
    '''
    def step(self, external_input=None, noise_std=0.0):
        # compute output
        x = float(self.phi @ self.r)  # scalar output

        # recurrent + feedback drive
        J = self.w @ self.r + self.eta * x

        # add external input or noise
        drive = J.copy()
        if external_input is not None:
            # external_input shape should be (num_neurons,) or scalar
            drive += external_input
        if noise_std > 0.0:
            drive += np.random.randn(self.num_neurons) * noise_std

        # step neurons -> spikes
        spike = np.array([n.step(drive[i]) for i, n in enumerate(self.neurons)])

        # filtered traces: h (fast), r (slow)
        # Euler updates
        self.h += self.dt * (-self.h / self.tau_r + spike / self.tau_r)
        self.r += self.dt * (-self.r / self.tau_d + self.h / self.tau_d)

        # record
        self.spike_history.append(spike.copy())
        self.time_history.append(self.time)
        self.time += self.dt

        return x, spike
    '''
    def train_loss(self, target):
        self.P -= (self.P @ np.outer(self.r, self.r) @ self.P) / (1.0 + self.r.T @ self.P @ self.r)
        self.phi += (target - self.phi @ self.r) * (self.P @ self.r)

    def run(self, simtime, target=None, train=False):
        time_steps = int(simtime / self.dt)
        x_rec = np.zeros(time_steps)
        spike_rec = np.zeros((time_steps, self.num_neurons))
        for t in range(time_steps):
            print(f'time: {t * self.dt:.2f} ms', end='\r')
            x, spike = self.step(0.5 * np.random.randn(self.num_neurons) if (t < 10) else 0.0)
            x_rec[t] = x
            spike_rec[t] = spike
            if train and target is not None and t > 50.:
                self.train_loss(target[t])
        return x_rec, spike_rec
    

if __name__ == "__main__":
    simtime = 1000.0
    dt = 0.1
    time_steps = int(simtime / dt)
    t = np.arange(0, simtime, dt)
    target = np.sin(2 * mt.pi * 1 * t / 1000.0) + np.sin(2 * mt.pi * 3 * t / 1000.0)# + np.sin(2 * mt.pi * 5 * t / 1000.0)
    net = spikingNeuronNetwork(num_neurons=300, dt=dt, p=1.0, q=1.5, p_w=1., lambda_=1.0)
    x_rec, spike_rec = net.run(simtime, target=target, train=True)
    x_rec_trained, spike_rec_trained = net.run(simtime, train=False)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, target, label='Target', color='blue')
    plt.plot(t, x_rec, label='Output', color='red', alpha=0.7)
    plt.plot(t, x_rec_trained, label='Output (after training)', color='green', alpha=0.7)
    plt.title('Target vs Output')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.imshow(spike_rec.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net.num_neurons])
    plt.imshow(spike_rec_trained.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net.num_neurons])
    plt.title('Spike Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')

    plt.tight_layout()
    plt.show()