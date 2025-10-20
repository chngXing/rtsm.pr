import numpy as np
import matplotlib.pyplot as plt

def cross_correlogram(t_ref, t_tgt, bin_size=1.0, window=100.0):
    bins = np.arange(-window, window + bin_size, bin_size)
    dt = []

    for tr in t_ref:
        dt.extend(t_tgt - tr)

    dt = np.array(dt)
    cc, edges = np.histogram(dt, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    plt.bar(centers, cc, width=bin_size)
    plt.xlabel("Time difference (ms)")
    plt.ylabel("Count")
    plt.title("Cross-Correlogram")
    plt.show()

# Example
t_ref = np.array([10, 25, 50, 80, 120, 180])
t_tgt = np.array([12, 27, 55, 83, 130, 200])
cross_correlogram(t_ref, t_tgt)
