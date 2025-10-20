# lif_force_torch_gpu.py
import torch
import matplotlib.pyplot as plt
import math as mt

# ---------- LIF Neuron ------------
class LIFNeuron(torch.nn.Module):
    def __init__(self, dt=0.1, tau_m=10.0, v_th=1.0, v_reset=0.0, I_bias=0.3):
        super().__init__()
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.I_bias = I_bias
        self.v = None
        self.spike = None

    def reset_state(self, N, device):
        self.v = torch.zeros(N, device=device)
        self.spike = torch.zeros(N, device=device)

    def forward(self, input_current):
        dv = (-self.v / self.tau_m + input_current + self.I_bias) * self.dt
        self.v += dv
        spiked = (self.v >= self.v_th).float()
        self.v = torch.where(spiked.bool(), torch.tensor(self.v_reset, device=self.v.device), self.v)
        self.spike = spiked
        return spiked


# ---------- Spiking Reservoir + FORCE ------------
class SpikingReservoirFORCE:
    def __init__(self, N=300, dt=0.1, tau_r=2.0, tau_d=20.0, q=1.5, p=1.0,
                 w_scale=0.05, I_bias=0.3, lambda_=1.0, seed=42, device='cuda'):
        torch.manual_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.N = N
        self.dt = dt
        self.tau_r = tau_r
        self.tau_d = tau_d
        self.q = q
        self.p = p

        # weights
        self.W = torch.randn(N, N, device=self.device) * w_scale
        self.eta = torch.randn(N, device=self.device) * 0.5  # feedback
        self.phi = torch.zeros(N, device=self.device)
        self.P = torch.eye(N, device=self.device) / lambda_

        # states
        self.h = torch.zeros(N, device=self.device)
        self.r = torch.zeros(N, device=self.device)
        self.neuron = LIFNeuron(dt=dt, I_bias=I_bias)
        self.neuron.reset_state(N, self.device)

    def step(self, noise_std=0.0):
        x = torch.dot(self.phi, self.r)
        J = self.q * self.W @ self.r + self.p * self.eta * x
        if noise_std > 0:
            J += torch.randn_like(J) * noise_std
        spike = self.neuron(J)
        self.h += self.dt * (-self.h / self.tau_r + spike / self.tau_r)
        self.r += self.dt * (-self.r / self.tau_d + self.h / self.tau_d)
        return x.item(), spike

    def rls_update(self, target):
        r = self.r.unsqueeze(1)
        Pr = self.P @ r
        denom = 1.0 + torch.dot(r.squeeze(), Pr.squeeze())
        k = Pr / denom
        y_hat = torch.dot(self.phi, self.r)
        error = target - y_hat
        self.phi += (k.squeeze() * error)
        self.P -= k @ Pr.T
        return y_hat.item(), error.item()

    def run(self, simtime, target=None, train=False, noise_std=0.5, noise_duration=200):
        steps = int(simtime / self.dt)
        x_rec = torch.zeros(steps, device=self.device)
        spike_rec = torch.zeros(steps, self.N, device=self.device)
        err_rec = torch.zeros(steps, device=self.device) if train else None

        for t in range(steps):
            noise = noise_std if (t * self.dt) < noise_duration else 0.0
            x, spike = self.step(noise_std=noise)
            x_rec[t] = x
            spike_rec[t] = spike
            if train and target is not None:
                y_hat, e = self.rls_update(torch.tensor(target[t], device=self.device))
                err_rec[t] = e
        return x_rec.cpu().numpy(), spike_rec.cpu().numpy(), err_rec.cpu().numpy() if err_rec is not None else None


# ---------- Main ----------
if __name__ == "__main__":
    simtime = 2000.0
    dt = 0.1
    t = torch.arange(0, simtime, dt)
    target = (torch.sin(2 * mt.pi * 1 * t / 1000.0)
             + 0.5 * torch.sin(2 * mt.pi * 3 * t / 1000.0)).numpy()

    net = SpikingReservoirFORCE(N=300, dt=dt, I_bias=0.3, device='cuda')
    x_rec, spike_rec, err = net.run(simtime, target=target, train=True, noise_std=0.8, noise_duration=200)
    x_trained, spike_trained, err_trained = net.run(simtime, train=False, noise_std=0.0)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, target, label='Target')
    plt.plot(t, x_rec, label='Output')
    plt.plot(t, x_trained, label='Output (after training)', linestyle='--')
    plt.legend()
    plt.title('Target vs Output (GPU)')

    plt.subplot(3, 1, 2)
    plt.imshow(spike_rec.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net.N])
    plt.imshow(spike_trained.T, aspect='auto', cmap='gray_r', extent=[0, simtime, 0, net.N])
    plt.ylabel('Neuron index')
    plt.title('Spike raster')

    plt.subplot(3, 1, 3)
    if err is not None:
        plt.plot(t, err, label='RLS error')
        plt.legend()
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()
