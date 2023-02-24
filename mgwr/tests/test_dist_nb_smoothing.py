"""
outdated
"""

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

import numpy as NUM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf as pdf

x, y, coords, data_path = get_georgia_data()
test_case = MGWR(x, y, coords, data_path, neighbor_type="neighbor", kernel="bisquare")

distance_, distance_smooth, distance_gradient = [], [], []

for j in np.arange(test_case.n):
    distances_from_j = np.sqrt(np.add(test_case.coords_scaled[j, 0], -test_case.coords_scaled[:, 0].T) ** 2 +
                         np.add(test_case.coords_scaled[j, 1], -test_case.coords_scaled[:, 1].T) ** 2)
    distances_from_j.sort()
    poly_coef_j = test_case.poly_coef[j]

    d_, d_s, d_g = [], [], []
    for bw in np.arange(test_case.lb_neighbors_, test_case.n, 5):
        print(bw)
        d_.append(distances_from_j[bw])
        bw = np.asarray(bw) * 1.00
        d_s.append(np.sum(poly_coef_j * np.power(bw, seq1)))
        d_g.append(np.sum(poly_coef_j[:-1] * (seq2 * np.power(bw, seq3))))

    distance_.append(d_)
    distance_smooth.append(d_s)
    distance_gradient.append(d_g)

distance_ = NUM.asarray(distance_)
distance_smooth = NUM.asarray(distance_smooth)
distance_gradient = NUM.asarray(distance_gradient)


### distance-neighbors smoothing
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
fig.tight_layout(pad=2)

iplot = 0
for i in range(test_case.n):
    axs[iplot].plot(np.arange(test_case.lb_neighbors_, test_case.n, 5), distance_[i, :], alpha=0.3, c="grey")
    axs[iplot].set_xlabel("Number of neighbors", fontsize=16)
    axs[iplot].set_ylabel("Distance", fontsize=16)
    axs[iplot].set_title("a", loc="left", fontsize=16)

iplot = 1
for i in range(test_case.n):
    axs[iplot].plot(np.arange(test_case.lb_neighbors_, test_case.n, 5), distance_smooth[i, :], alpha=0.3, c="grey")
    axs[iplot].set_xlabel("Number of neighbors", fontsize=16)
    axs[iplot].set_ylabel("Distance Smoothed", fontsize=16)
    axs[iplot].set_title("b", loc="left", fontsize=16)

fig.savefig(os.path.join(os.path.dirname(__file__), r"test\test_output\dist_nb_smoothing_temp.png"))
plt.close()

