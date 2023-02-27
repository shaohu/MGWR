### parallel in gradient_mgwr
### split xxx with njit
### slow

from kernels import *
import numpy as NUM
import numpy as np
import time
from numpy.linalg import inv as inv
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from line_profiler import LineProfiler
from math import sqrt
from numba import njit, jit, prange

def opt_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    aicc, daicc, _ = gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef)
    return aicc, daicc

def hess_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    _, _, d2aicc = gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, True, poly_coef)
    return d2aicc

def opt_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    aicc, daicc, _ = gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef)
    return aicc, daicc.flatten()

def hess_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    _, _, d2aicc = gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, True, poly_coef)
    return d2aicc

def gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef, test=False,
                 finalize=False):
    """
    this function read in data and bandwidth; returns AICc, gradient of AICc and all f(xi) components
    :param banwidth: a list of bandwidths
    :param y_data: y of shape (n,1)
    :param x_data: x of shape (n,k) including the intercept
    :param neighbor_type: "distance" or "neighbor"
    :param kernel: "gaussian" or "bisquare"
    :param hess: boolean, use Hessian matrix information or not
    :param poly_coef: the polynomial coefficients of distance-neighbors of shape for example (n, 5)
    :param test: boolean, allow function to have other returns if True, for testing purpose
    :return aicc, daicc, d2aicc
    """
    n, k = x_data.shape

    ### final output
    y_hat = NUM.zeros((n, 1), dtype=float)
    if finalize:
        beta = NUM.zeros((n, k), dtype=float)
        std_beta = NUM.zeros((n, k), dtype=float)
        influence = NUM.zeros((n, 1), dtype=float)
        ENP = 0.

    trS = 0.

    ### used in loop for local fit
    XtW = NUM.zeros((k, n), dtype=float)
    XtWX_inv = NUM.zeros((k, k), dtype=float)
    XtWX_inv_XtW = NUM.zeros((k, n), dtype=float)
    XtWX_inv_XtWD = NUM.zeros((k, n), dtype=float)

    # if hess:
    #     XtWX_inv_XtWDX = NUM.zeros((k, k), dtype=float)
    #     XtWX_inv_XtWDX_XtWX_inv = NUM.zeros((k, k), dtype=float)
    #     XtWX_inv_XtWDD = NUM.zeros((k, n), dtype=float)

    y_hat_local = NUM.zeros((n, 1), dtype=float)

    dRSS = 0.
    dtrS = 0.

    # if hess:
    #     d2trS = 0.
    #     d2RSS = 0.
    #     d2RSS_all = 0.
    #     d2RSS_diag = 0.

    for i in range(n):

        # TODO https://stackoverflow.com/questions/37298710/fastest-way-to-compute-distance-beetween-each-points-in-python
        distances = NUM.sqrt(NUM.add(coords_scaled[i, [0]], -coords_scaled[:, [0]].T) ** 2 +
                             NUM.add(coords_scaled[i, [1]], -coords_scaled[:, [1]].T) ** 2)

        w, wd, wdd = kernel_func(distances, banwidth, hess, poly_coef[i, :], n)

        # w = w.reshape((1, -1))
        # wd = wd.reshape((1, -1))
        # if hess:
        #     wdd = wdd.reshape((1, -1))

        XtW = x_data.T * w
        XtWX = XtW.dot(x_data)
        XtWX_inv = inv(XtWX)
        XtWX_inv_XtW = XtWX_inv.dot(XtW)
        XtWX_inv_XtWD = XtWX_inv.dot(x_data.T) * wd  ### AiBi in (24)(25)

        # if hess:
        #     XtWX_inv_XtWDX = XtWX_inv_XtWD.dot(x_data)
        #     XtWX_inv_XtWDX_XtWX_inv = XtWX_inv_XtWDX.dot(XtWX_inv)
        #     XtWX_inv_XtWDD = XtWX_inv.dot(x_data.T) * wdd

        trS += x_data[i].dot(XtWX_inv_XtW[:, [i]])

        beta_all = XtWX_inv_XtW.dot(y_data)

        if finalize:
            influence[i] = x_data[i].dot(XtWX_inv_XtW[:, [i]])
            ENP += influence[i]  # TODO haven't check this
            beta[i] = beta_all.T
            std_beta[i] = NUM.sqrt(NUM.diag(XtWX_inv_XtW.dot(XtWX_inv_XtW.T)))  # TODO haven't check this # TODO sqrt

        y_hat_local = x_data.dot(beta_all)
        y_hat[i] = x_data[i].dot(beta_all)

        d_yhat_i = x_data[i].dot(XtWX_inv_XtWD).dot(y_data - y_hat_local)  # 1k*kn*n1 = O(kn)
        dRSS += - 2 * d_yhat_i * (y_data[i] - y_hat[i])

        # if hess:
        #     d2_yhat_i = x_data[i].dot(-2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWD) + \
        #                               XtWX_inv_XtWDD).dot(y_data - y_hat_local)  # 1k * (kk*kn + kn*n1) = O(k2n)
        #     d2RSS_all += 2 * (d_yhat_i ** 2)
        #     d2RSS_diag += - 2 * d2_yhat_i * (y_data[i] - y_hat[i])

        dtrS += - x_data[i].dot(XtWX_inv_XtWD).dot(x_data).dot(XtWX_inv).dot(x_data[i])  # 1k*kn*nk*k1 = O(nk)

        # if hess:
        #     d2trS += - x_data[i].dot(- 2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWDX_XtWX_inv) + \
        #                              XtWX_inv_XtWDD.dot(x_data).dot(XtWX_inv)).dot(x_data[i])  ### ...(28)
        # # 1k * (kk*kk + kn*nk*kk)*k1 = O(k2n)

    ### get fitted value, trace(S), RSS, aicc
    trS = NUM.asarray(trS).item()

    RSS = NUM.sum((y_data - y_hat) ** 2)
    aicc = n * NUM.log(RSS / n) + n * NUM.log(2.0 * NUM.pi) + n * (trS + n) / (n - trS - 2.0)

    if finalize:
        sigma = NUM.sqrt(RSS / (n - trS)) # TODO sqrt
        std_beta = std_beta * sigma

    ### Jacobian, Hessian of RSS
    # if hess:
    #     d2RSS = d2RSS_all + d2RSS_diag

    ### Jacobian, Hessian of aicc
    daicc = n / RSS * dRSS + 2 * n * (n - 1) / ((n - trS - 2) ** 2) * dtrS
    daicc = NUM.asarray(daicc).item()

    # d2aicc = None
    # if hess:
    #     d2aicc_rss = - n / RSS ** 2 * dRSS ** 2 + n / RSS * d2RSS  ### ...(18)
    #     d2aicc_trs = 4 * n * (n - 1) / ((n - trS - 2) ** 3) * (dtrS ** 2) + 2 * n * (n - 1) / (
    #             (n - trS - 2) ** 2) * d2trS  ### ...(27)
    #     d2aicc = d2aicc_rss + d2aicc_trs  ### ...(17)

    localR2 = None
    if finalize:
        return beta, std_beta, y_hat, trS, ENP, RSS, aicc, sigma, influence, localR2

    # if test:
    #     if hess:
    #         return aicc, daicc, d2aicc, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS
    #     else:
    #         return aicc, daicc, RSS, dRSS, trS, dtrS
    # if hess:
    #     return aicc, daicc, d2aicc
    return aicc, daicc, None

    # dy_hat = NUM.zeros((n, 1), dtype=float)
    # d2y_hat = NUM.zeros((n, 1), dtype=float)
    # S = NUM.zeros((n, n), dtype=float)
    # dtrSi = NUM.zeros(n, dtype=float)
    # d2trSi = NUM.zeros((n, 1), dtype=float)
    # for i in range(n):
    #   S[i, :] = x_data[i].dot(XtWX_inv_XtW)
    #   y_hat_local = NUM.sum(x_data * beta[i, :], axis=1).reshape((-1, 1))
    #   dtrSi[i] = - x_data[i].dot(XtWX_inv_XtWD).dot(x_data).dot(XtWX_inv).dot(x_data[i])  ### ...(20)
    #   d2trSi[i] = - x_data[i].dot(- 2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWDX_XtWX_inv) + \
    #                               XtWX_inv_XtWDD.dot(x_data).dot(XtWX_inv)).dot(x_data[i])  ### ...(28)
    #   dy_hat[i] = x_data[i].dot(XtWX_inv_XtWD).dot(y_data - y_hat_local)                  ### ...(18)
    #   d2y_hat[i] = x_data[i].dot(-2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWD) + \
    #                            XtWX_inv_XtWDD).dot(y_data - y_hat_local)            ### ...(26)
    # d2RSS_all = 2 * dy_hat.T.dot(dy_hat)
    # d2RSS_diag = - 2 * NUM.sum(d2y_hat * (y_data - y_hat), axis=0)
    # d2RSS = d2RSS_all + NUM.diag(d2RSS_diag)                                    ### ...(19)
    # same as d2RSS = 2 * NUM.sum(dy_hat ** 2 - (y_data - y_hat) * d2y_hat)     ### ...(19)
    # dRSS = - 2 * dy_hat.T.dot(y_data - y_hat)                                       ### ...after (22)
    # y_hat = S.dot(y_data)
    # trS = S.trace()
    # dtrS = NUM.sum(dtrSi)                                                           ### ...(20)
    # d2trS = NUM.sum(d2trSi)                                                     ### ...(28)

@njit
def xxx(x_data, y_data, w, wd, j):
    XtW = x_data.T * w # *(kn)(n)
    XtWX = XtW.dot(x_data) # (kn)(nk)
    XtWX_inv = inv(XtWX) # (kk inv)

    XtWX_inv_Xt = XtWX_inv.dot(x_data.T) # (kk)(kn)
    XtWX_inv_XtW = XtWX_inv.dot(XtW) # (kk)(kn)
    XtWX_inv_XtW_row_j = XtWX_inv_XtW[j, :]
    XtWX_inv_XtWD_row_j = XtWX_inv_Xt[j, :] * wd # (n)(n) = shape(n, )
    XtWX_inv_XtWDX_row_j = XtWX_inv_XtWD_row_j.dot(x_data) # (n)(nk) = shape(k, )
    beta_all = XtWX_inv_XtW.dot(y_data) # (kn)(n1)
    y_hat_local = x_data.dot(beta_all)
    beta_ij = beta_all[j, 0]
    return XtWX_inv, XtWX_inv_XtW_row_j, XtWX_inv_XtWD_row_j, XtWX_inv_XtWDX_row_j, y_hat_local, beta_ij

@njit
def lloop(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, nlow, nhigh, n, k):
    output_size = (nhigh-nlow) * k
    output_list = np.zeros((output_size, 9+2))
    ## 2 col for i, j, 9 for output

    for i in range(nlow, nhigh):
        poly_coef_i = poly_coef[i, :] # .reshape((1, -1))

        for j in range(k):
            banwidth_j = banwidth[j]

            output_i = (i - nlow) * k + j

            ################# the double-for-loop that could be paralleled #################
            # TODO https://stackoverflow.com/questions/37298710/fastest-way-to-compute-distance-beetween-each-points-in-python
            distances = NUM.sqrt(NUM.add(coords_scaled[i, 0], -coords_scaled[:, 0].T) ** 2 +
                                 NUM.add(coords_scaled[i, 1], -coords_scaled[:, 1].T) ** 2) # distances shape(n, )

            w, wd, wdd = kernel_func(distances, banwidth_j, hess, poly_coef_i, n)

            #####################
            XtW = x_data.T * w # *(kn)(n)
            XtWX = XtW.dot(x_data) # (kn)(nk)
            XtWX_inv = inv(XtWX) # (kk inv)

            XtWX_inv_Xt = XtWX_inv.dot(x_data.T) # (kk)(kn)
            XtWX_inv_XtW = XtWX_inv.dot(XtW) # (kk)(kn)
            XtWX_inv_XtW_row_j = XtWX_inv_XtW[j, :]
            XtWX_inv_XtWD_row_j = XtWX_inv_Xt[j, :] * wd # (n)(n) = shape(n, )
            XtWX_inv_XtWDX_row_j = XtWX_inv_XtWD_row_j.dot(x_data) # (n)(nk) = shape(k, )
            beta_all = XtWX_inv_XtW.dot(y_data) # (kn)(n1)
            y_hat_local = x_data.dot(beta_all)
            ### if finalize:
            beta_ij = beta_all[j, 0]
            ### if finalize:
            #####################

            dy_hat_ij = x_data[i, j] * XtWX_inv_XtWD_row_j.dot(y_data - y_hat_local)[0]

            dtrS_j = - x_data[i, j] * XtWX_inv_XtWDX_row_j.dot(XtWX_inv).dot(x_data[i, :])
                                        # k1 kk k1

            Ri_j = x_data[i, j] * XtWX_inv_XtW_row_j
            Riij = Ri_j[i]
            Ri_j_sq = np.sqrt(Ri_j.dot(Ri_j.T))

            y_hat_i = Ri_j.dot(y_data)[0] # y_hat_local[i]? TODO

            trS_ij = x_data[i, j] * XtWX_inv_XtW_row_j[i]

            output_i = (i - nlow) * k + j
            output_list[output_i, 0] = i
            output_list[output_i, 1] = j
            output_list[output_i, 2:] = np.asarray([beta_ij, dy_hat_ij, dtrS_j, 0., trS_ij, Riij, y_hat_i, Ri_j_sq, 0.])

    return output_list


# @profile
def gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef=None, test=False,
                  finalize=False):
    """
    this function read in data and bandwidth; returns AICc, gradient of AICc and all f(xi) components
    :param banwidth: a list of bandwidths
    :param y_data: y of shape (n,1)
    :param x_data: x of shape (n,k) including the intercept
    :param neighbor_type: "distance" or "neighbor"
    :param kernel: "gaussian" or "bisquare"
    :param hess: boolean, use Hessian matrix information or not
    :param poly_coef: the polynomial coefficients of distance-neighbors of shape for example (n, 5)
    :param test: boolean, allow function to have other returns if True, for testing purpose
    :return aicc, daicc, d2aicc
    """
    n, k = x_data.shape
    print("in gradient_mgwr bw ", banwidth)

    ################# final output #################
    if finalize:
        localR2 = None  # TODO
        RSS = None
        aicc = None
        sigma = None

    # if hess:
    #     d2trS = NUM.zeros((k, 1), dtype=float)  ### additive
    #     d2y_hat = NUM.zeros((n, k), dtype=float)  ### additive # TODO can remove, but tricky one
    print("start loops")

    num_cores = 10
    step = int(np.ceil(n / num_cores))
    ub = step * np.arange(1, num_cores + 1)
    lb = step * np.arange(0, num_cores)
    if ub[-1] > n:
        ub[-1] = n

    pool = mp.Pool(num_cores)
    ### TODO in all process, cache x_data,
    param_list = []
    for i in range(num_cores):
        param_list.append([poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, lb[i], ub[i], n, k])

    ### start multiprocessing
    results = [pool.apply_async(lloop, args=param) for param in param_list]
    pool.close()
    results = np.vstack([p.get() for p in results])
    results = results[np.argsort(results[:, 1])]  # k
    results = results[np.argsort(results[:, 0])]  # n

    beta_ij = results[:, 2].reshape((n, k))
    beta = beta_ij
    dy_hat_ij = results[:, 3].reshape((n, k))
    dy_hat = dy_hat_ij
    dtrS_j = results[:, 4].reshape((n, k))
    dtrS = dtrS_j.sum(axis=0).reshape(k, 1)
    trS_ij = results[:, 6].reshape((n, k)).sum(axis=0)
    trS = sum(trS_ij)
    Riij = results[:, 7].reshape((n, k))
    ENPj = Riij.sum(axis=0)
    influence = Riij.sum(axis=1).reshape((n, 1))
    y_hat_i = results[:, 8].reshape((n, k))
    y_hat = y_hat_i.sum(axis=1).reshape((n, 1))
    Ri_j_sq = results[:, 9].reshape((n, k))
    std_beta = Ri_j_sq * NUM.abs(1. / x_data)

    ################# final output #################
    daicc = NUM.zeros((k, 1), dtype=float)
    d2aicc = NUM.zeros((k, k), dtype=float)
    ################# final output #################

    ################# intermediate #################
    dRSS = NUM.zeros((k, 1), dtype=float)
    # if hess:
    #     d2RSS_all = NUM.zeros((k, k), dtype=float)
    #     d2RSS = NUM.zeros((k, k), dtype=float)
    #     d2aicc_rss = NUM.zeros((k, k), dtype=float)
    #     d2aicc_trs = NUM.zeros((k, k), dtype=float)
    ################# intermediate #################

    ################# get output #################
    trS = NUM.asarray(trS).item()

    RSS = NUM.sum((y_data - y_hat) ** 2)
    aicc = n * NUM.log(RSS / n) + n * NUM.log(2.0 * NUM.pi) + n * (trS + n) / (n - trS - 2.0)
    sigma = NUM.sqrt(RSS / (n - trS)) # TODO sqrt

    if finalize:
        std_beta = std_beta * sigma
    ################# get output #################

    ################# Jacobian, Hessian of RSS #################
    dRSS = -2 * dy_hat.T.dot(y_data - y_hat)
    # if hess:
    #     d2RSS_all = 2 * dy_hat.T.dot(dy_hat)
    #     d2RSS_diag = NUM.sum(d2y_hat * (y_data - y_hat), axis=0)
    #     d2RSS = d2RSS_all + NUM.diag(d2RSS_diag)

    ################# Jacobian, Hessian of RSS #################

    ################# Jacobian, Hessian of aicc #################
    print(RSS, dRSS.shape, trS, dtrS.shape)
    daicc = n / RSS * dRSS + 2 * n * (n - 1) / ((n - trS - 2) ** 2) * dtrS
    # if hess:
    #     d2aicc_rss = - n / RSS ** 2 * dRSS.dot(dRSS.T) + n / RSS * d2RSS
    #     d2aicc_trs = 4 * n * (n - 1) / ((n - trS - 2) ** 3) * dtrS.dot(dtrS.T) + 2 * n * (n - 1) / (
    #                 (n - trS - 2) ** 2) * NUM.diag(d2trS)
    #     d2aicc = d2aicc_rss + d2aicc_trs
    ################# Jacobian, Hessian of aicc #################
    print("in gradient_mgwr end")
    if finalize:
        return beta, std_beta, y_hat, trS, ENPj, RSS, aicc, sigma, influence, localR2
    # if test:
        # if hess:
        #     return aicc, daicc, d2aicc, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS
        # else:
        #     return aicc, daicc, RSS, dRSS, trS, dtrS
    return aicc, daicc, d2aicc


########################################################################################################################
## lloop, nogil=True 21s
## lloop, nogil=True,parallel=True 73s
## lloop, nogil=True,fastmath=True 21s
## SVML not helping
## prange not helping (but loops not paralleled anyway)
#
# @njit(nogil=True)
# def test_func(x_data, y_data, coords_scaled, poly_coef, kernel_func, n, k):
#     s = np.zeros(k)
#     for i in range(n):
#         poly_coef_i = poly_coef[i, :] # .reshape((1, -1))
#         for j in range(k):
#             distances = NUM.sqrt(NUM.add(coords_scaled[i, 0], -coords_scaled[:, 0].T) ** 2 +
#                                  NUM.add(coords_scaled[i, 1], -coords_scaled[:, 1].T) ** 2) # distances shape(n, )
#             banwidth_j = banwidth[j]
#             w, wd, wdd = kernel_func(distances, banwidth_j, hess, poly_coef_i, n)
#             XtW = x_data.T * w # *(kn)(n)
#             XtWX = XtW.dot(x_data) # (kn)(nk)
#             XtWX_inv = inv(XtWX) # (kk inv)
#
#             XtWX_inv_Xt = XtWX_inv.dot(x_data.T) # (kk)(kn)
#             XtWX_inv_XtW = XtWX_inv.dot(XtW) # (kk)(kn)
#             XtWX_inv_XtWD_row_j = XtWX_inv_Xt[j, :] * wd # (n)(n) = shape(n, )
#             XtWX_inv_XtWDX_row_j = XtWX_inv_XtWD_row_j.dot(x_data) # (n)(nk) = shape(k, )
#
#             # beta_all = XtWX_inv_XtW.dot(y_data) # (kn)(n1)
#             #
#             # ### if finalize:
#             # beta_ij = beta_all[j, 0]
#             # ### if finalize:
#             #
#             # y_hat_local = x_data.dot(beta_all)
#             #
#             # dy_hat_ij = x_data[i, j] * XtWX_inv_XtWD_row_j.dot(y_data - y_hat_local)[0]
#             #
#             # dtrS_j = - x_data[i, j] * XtWX_inv_XtWDX_row_j.dot(XtWX_inv).dot(x_data[i, :])
#             #
#             # Ri_j = x_data[i, j] * XtWX_inv_XtW[j, :]
#             # Riij = Ri_j[i]
#             # Ri_j_sq = np.sqrt(Ri_j.dot(Ri_j.T))
#             #
#             # y_hat_i = Ri_j.dot(y_data)[0] # y_hat_local[i]? TODO
#             #
#             # trS_ij = x_data[i, j] * XtWX_inv_XtW[j, i]
#
#         s += XtWX_inv_XtWDX_row_j
#     return s
#

import os
import pandas as pd
import numpy as NUM
import numpy as np
import sys
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

import time
from kernels import *
if __name__ == '__main__':
        n = 5000
        k = 11
        np.random.seed(1)
        x_data = np.random.random((n, k))
        y_data = np.random.random((n, 1))
        coords_scaled = np.random.random((n, 2))
        poly_coef = np.random.random((n, 6))#.astype(float)
        banwidth = np.random.randint(0, n, k)#.astype(np.int64)

        hess = False
        neighbor_type = "neighbors"
        kernel_func = kernel_nb_bisq
        finalize = False

        # njit_time = 0
        # start = time.time()
        # output_list = lloop(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, 0, n, n, k)
        #
        # end = time.time()
        # njit_time += end - start
        # print("time ", njit_time)
        # print(output_list.shape)
        # lloop.inspect_types()
        # lloop.parallel_diagnostics(level=4)


        start = time.time()
        output_list = gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef=poly_coef,
                      test=False,
                      finalize=False)
        print(output_list)

        end = time.time()
        print("time ", end - start)



        #
        # njit_time = 0
        # start = time.time()
        # s = test_func(x_data, y_data, coords_scaled, poly_coef, kernel_func, n, k)
        # print(s)
        # end = time.time()
        # njit_time += end - start
        # print("time ", njit_time)
        # print(test_func.inspect_types())
        # lloop.parallel_diagnostics(level=4)

### before loop-loop
# fx = NUM.zeros((n, k), dtype=float)
# S = NUM.zeros((n, n), dtype=float)
# R = NUM.zeros((n, n, k), dtype=float)
# dtrSi = NUM.zeros((n, k), dtype=float)
# d2trSi = NUM.zeros((n, k), dtype=float)
# d2_yhat = NUM.zeros((n, k), dtype=float) ### additive
### loop-loop
# ej = NUM.zeros((1, k), dtype=float)
# ej[:, j] = 1
# equivalent to
# R[i, :, j] = x_data[i, [j]] * ej.dot(XtWX_inv_XtW)
# equivalent to
# dy_hat[i, j] = x_data[i, [j]] * ej.dot(XtWX_inv_XtWD).dot(y_data - y_hat_local) # ...................(40)
# dy_hat[i, j] = x_data[i, [j]] * XtWX_inv_XtWD.dot(y_data - y_hat_local)[j]  # ...................(40)
# equivalent to
#     d2_yhat[i, j] = 2 * x_data[i, [j]] * (ej.dot(XtWX_inv_XtWDX).dot(XtWX_inv_XtWD).dot(y_data - y_hat_local) \
#                                         - ej.dot(XtWX_inv_XtWDD).dot(y_data - y_hat_local) \
#                                         - ej.dot(XtWX_inv_XtWD).dot(y_data - dy_hat_local))
# equivalent to
#     d2_yhat[i, j] = 2 * x_data[i, [j]] * (XtWX_inv_XtWDX.dot(XtWX_inv_XtWD).dot(y_data - y_hat_local)[j] \
#                                          - XtWX_inv_XtWDD.dot(y_data - y_hat_local)[j] \
#                                          - XtWX_inv_XtWD.dot(y_data - dy_hat_local)[j])
# equilvalent to
#     dtrSi[i, j] = - x_data[i, [j]] * XtWX_inv_XtWDX[[j], :].dot(XtWX_inv).dot(x_data[i, :]) # .......(41)
# equivalent to
#     d2trSi[i, j] = - x_data[i, [j]] * (-2 * XtWX_inv_XtWDX[[j], :].dot(XtWX_inv_XtWDX_XtWX_inv) + \
#                                         XtWX_inv_XtWDD[[j], :].dot(x_data).dot(XtWX_inv)).dot(x_data[i, :])
# equivalent to
#     d2trSi[i, j] = - x_data[i, [j]] * (-2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWDX_XtWX_inv) + \
#                     XtWX_inv_XtWDD.dot(x_data).dot(XtWX_inv)).dot(x_data[i, :])[j]
# equivalent to
#     dtrSi[i, j] = - x_data[i, [j]] * (XtWX_inv_XtWD.dot(x_data).dot(XtWX_inv)[j,: ]).dot(x_data.T[:, i])
# note
# assert NUM.all(XtWX_inv_XtWDX_XtWX_inv == (XtWX_inv.dot(x_data.T) * wd).dot(x_data).dot(XtWX_inv))
# assert NUM.all(XtWX_inv_XtW[[j], :] - XtWX_inv_Xt[[j], :] * w)
# assert NUM.all(XtWX_inv_XtWD[[j], :] == XtWX_inv_Xt[[j], :] * wd)
# assert NUM.all(XtWX_inv_XtWDX[[j], :] == XtWX_inv_XtWD[[j], :].dot(x_data))


# beta[i, j] = beta_all[j]
# dy_hat[i, j] = x_data[i, [j]] * XtWX_inv_XtWD[[j], :].dot(y_data - y_hat_local) # ...................(40)
# dtrS[j] += - x_data[i, [j]] * XtWX_inv_XtWDX[[j], :].dot(XtWX_inv).dot(x_data[i, :]) # .......(41)
# d2_yhat[i, j] = 2 * x_data[i, [j]] * (temp1 - temp2 - temp3)
# d2trS[j, j] += - x_data[i, [j]] * (-2 * XtWX_inv_XtWDX[[j], :].dot(XtWX_inv_XtWDX_XtWX_inv) + \
#                                         XtWX_inv_XtWDD[[j], :].dot(x_data).dot(XtWX_inv)).dot(x_data[i, :])
# trS += x_data[i, [j]] * XtWX_inv_XtW[j, i]
# d2RSS_diag = NUM.sum(d2_yhat * (y_data - y_hat), axis=0)

# fx[i, j] += Ri_j.dot(y_data)
# equivalent to
# Ri_j = R[i, :, j]
# R[i, :, j] = x_data[i, [j]] * XtWX_inv_XtW[j, :]

# equivalent to
# dy_hat_local = x_data.dot(XtWX_inv_XtWD).dot(y_data - y_hat_local) is O(kn2) slow
# equivalent to
# y_hat_local = x_data.dot(XtWX_inv_XtW).dot(y_data) is O(kn2) slow

### after loop-loop
# equivalent to
# S = NUM.sum(R, axis=-1)
# ENPj = R.trace()
# y_hat = S.dot(y_data)
# trS = S.trace()
# fx = NUM.zeros((n, k), dtype=float)
# for j in range(k):
#     fx[:,[j]] = R[:,:,j].dot(y_data)

### get local stats ###
# equivalent to
# C = NUM.zeros((n, n), dtype=float)
# std_coef = NUM.zeros((n, k), dtype=float)
# for j in range(k):
#     C = inv(NUM.diag(x_data[:, j]))
#     C = NUM.dot(C, R[:, :, j])
#     std_coef[:, j] = NUM.sqrt(NUM.diagonal(NUM.dot(C, C.T) * sigma ** 2))
### Jacobian, Hessian of trace(S)
# dtrS = dtrSi.sum(axis=0).reshape((-1, 1))
# if hess:
#     d2trS = NUM.diag(NUM.sum(d2trSi, axis=0))
