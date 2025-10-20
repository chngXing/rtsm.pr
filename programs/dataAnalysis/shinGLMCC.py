import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math as mt
from simulation.reseviorNetGPU import LIFNeurons, SpikingNeuronNetwork
from dataAnalysis.GLMCC import GLMCC
from torch import Tensor
import pandas as pd
import numpy as np

class ShinGLMCC(GLMCC):
    def __init__(self, bin_width=10.0, window=1000.0, delay=0.0, tau=10.0, beta=1.0, theta=None, device='cuda'):
        super(ShinGLMCC, self).__init__(bin_width=bin_width, window=window, delay=delay, tau=tau, beta=beta, theta=theta, device=device)
    
    @torch.no_grad()
    def _log_posterior(self, t_sp: Tensor, cc: Tensor) -> Tensor:
        theta_main = self.theta[: self.m]
        n_segments = 2
        bin_edges = torch.linspace(0, self.m, n_segments + 1, dtype=torch.long)
        beta_list = [-self.beta, 1.0]

        smooth_penalty = 0.0
        for i in range(n_segments):
            beta_i = beta_list[i]
            start = bin_edges[i]
            end = bin_edges[i + 1]
            diff_seg = self.theta[start:end-1] - self.theta[start+1:end]
            smooth_penalty += beta_i / (2 * self.delta) * torch.sum(diff_seg**2)

        log_posterior = (
            torch.dot(cc, theta_main)
            + torch.sum(self.theta[-2] * self.func_f(t_sp))
            + torch.sum(self.theta[-1] * self.func_f(-t_sp))
            - torch.sum(self.gk())
            - smooth_penalty
        )
        return log_posterior
    
if __name__ == "__main__":
    '''
    #df = pd.read_csv("dataAnalysis/example/sample_data.csv")
    #tensor_data = torch.as_tensor(df.to_numpy(dtype="float64"), dtype=torch.float64)
    #print(tensor_data.shape)
    net = SpikingNeuronNetwork(num_neurons=3000, delay=0.0)
    t = np.arange(0, 2000, 0.1)
    target = np.sin(2 * mt.pi * 0.8 * t / 200.0) + np.sin(2 * mt.pi * 1.2 * t / 200.0)
    x, spike_record = net.run(simtime=2000.0, target=target, train=True)
    spike_record = spike_record.squeeze().T
    plt.subplot(3, 1, 2)
    plt.imshow(spike_record.T, aspect='auto', cmap='gray_r', extent=[0, 2000, 0, net.num_neurons])
    plt.title('Spike Raster Plot (No Delay)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.show()

    glm = GLMCC(
        bin_width=10.0,
        window=60.0,
        delay=5.0,
        tau=20.0,
        beta=1e2,
        dtype=torch.float64,
    )

    tensor_data = glm.spike_time(t_i=spike_record[0][1000:], t_j=spike_record[1][1000:], dt=0.1)
    tensor_data = glm.spiketime_relative(spiketime_tar=tensor_data[0], spiketime_ref=tensor_data[1], window_size=60.0)

    print("Fitting shinGLMCC model...")
    glm.fit(tensor_data, clm=0.01, eta=0.5, max_iter=1000, verbose=True)

    glm.summary()

    fig, ax = plt.subplots(figsize=(10, 5))
    glm.plot(ax, tensor_data)
    ax.set_title("shinGLMCC Fit Example")
    plt.tight_layout()
    plt.show()
    '''
    df = pd.read_csv("dataAnalysis/example/sample_data.csv")
    tensor_data = torch.as_tensor(df.to_numpy(dtype="float64"), dtype=torch.float64)
    glm = GLMCC(delay=4.0)  # set synaptic delay to initialize GLMC

    fig, ax = plt.subplots(figsize=(3, 3))
    idx_i, idx_j = 6, 7

    # relative spiketime (target neuron - reference neuron)
    t_sp = glm.spiketime_relative(spiketime_tar=list(df.query('neuron==@idx_i').spiketime), 
                            spiketime_ref=list(df.query('neuron==@idx_j').spiketime), window_size=50.0)
    glm.plot(ax=ax, t_sp=t_sp)

    ax.set_title(f'cross-correlogram: neuron {idx_i} to {idx_j}')
    plt.show()

    def fit_and_plot(ax, idx_i, idx_j, delay, window_size=50.0, verbose=True):
        # prepare relative spiketime (target neuron - reference neuron)
        glm = GLMCC(delay=delay)  # tune synaptic delay
        t_sp = glm.spiketime_relative(spiketime_tar=list(df.query('neuron==@idx_i').spiketime), 
        spiketime_ref=list(df.query('neuron==@idx_j').spiketime), window_size=window_size)

        # model settings
        glm.fit(t_sp, verbose=verbose)
        glm.plot(ax=ax, t_sp=t_sp)

        # recommended plot layouts
        ax.set_xlabel(r'$\tau [ms]$', fontsize=16)
        ax.set_ylabel(r'$C(\tau)$', fontsize=16)
        ax.set_title(f'neuron {idx_i} to {idx_j}', fontsize=18)
        ax.tick_params(direction='in', which='major', labelsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        return glm
    
    fig, ax = plt.subplots(figsize=(4, 4))
    glm = fit_and_plot(ax=ax, idx_i=idx_i, idx_j=idx_j, delay=4.0)
    glm.summary()
    plt.show()