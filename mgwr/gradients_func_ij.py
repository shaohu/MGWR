### fast
### lloop has one i j

from kernels import *
import numpy as NUM
import numpy as np
import time
from numpy.linalg import inv as inv
from multiprocessing import Pool
from functools import partial
from line_profiler import LineProfiler
from math import sqrt
from numba import njit, jit

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
                             NUM.add(coords_scaled[i, [1]], -coords_scaled[:, [1]].T) ** 2).flatten()

        w, wd = kernel_func(distances, banwidth, poly_coef[i, :])

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

@njit(nogil=True)
def lloop(i, j, poly_coef_i, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k):
    # n, k = x_data.shape
    # i, j, poly_coef_i = *i_j_poly_coef_i

    d2trS_j = 0
    d2y_hat_ij = 0
    # beta_ij, dy_hat_ij, dtrS_j, d2trS_j, trS_ij, Riij, y_hat_i, Ri_j_sq, d2y_hat_ij = [0, ] * 9 # TODO?
    ################# used within the double-for-loop memory(5kn+4kk+7n) #################
    # w = NUM.zeros((n, 1), dtype=float)
    # wd = NUM.zeros((n, 1), dtype=float)
    # wdd = NUM.zeros((n, 1), dtype=float)
    # XtW = NUM.zeros((k, n), dtype=float)
    # XtWX = NUM.zeros((k, k), dtype=float)
    # XtWX_inv = NUM.zeros((k, k), dtype=float)
    # XtWX_inv_Xt = NUM.zeros((k, n), dtype=float)
    # XtWX_inv_XtW = NUM.zeros((k, n), dtype=float)  ## better not have this
    # XtWX_inv_XtWD = NUM.zeros((k, n), dtype=float)  ## better not have this
    # XtWX_inv_XtWDX = NUM.zeros((k, k), dtype=float)

    # beta_all = NUM.zeros((n, 1), dtype=float)
    # y_hat_local = NUM.zeros((n, 1), dtype=float)
    # if hess:
    #     beta_local_res = NUM.zeros((n, 1), dtype=float)

    # if hess:
    #     dy_hat_local = NUM.zeros((n, 1), dtype=float)
    #     XtWX_inv_XtWDD = NUM.zeros((k, n), dtype=float)  ## better not have this
    #     XtWX_inv_XtWDX_XtWX_inv = NUM.zeros((k, k), dtype=float)
    ################# used within the double-for-loop memory(5kn+4kk+7n) #################

    ################# the double-for-loop that could be paralleled #################
    # TODO https://stackoverflow.com/questions/37298710/fastest-way-to-compute-distance-beetween-each-points-in-python
    distances = NUM.sqrt(NUM.add(coords_scaled[i, 0], -coords_scaled[:, 0].T) ** 2 +
                         NUM.add(coords_scaled[i, 1], -coords_scaled[:, 1].T) ** 2) # distances shape(n, )
    ### flatten() failed in numba

    w, wd = kernel_func(distances, banwidth[j], poly_coef_i)

    # w = w.reshape((1, -1))
    # wd = wd.reshape((1, -1))
    # if hess:
    #     wdd = wdd.reshape((1, -1))

    XtW = x_data.T * w
    XtWX = XtW.dot(x_data)
    XtWX_inv = inv(XtWX)

    XtWX_inv_Xt = XtWX_inv.dot(x_data.T)
    XtWX_inv_XtW = XtWX_inv.dot(XtW)
    XtWX_inv_XtWD_row_j = XtWX_inv_Xt[j, :] * wd # shape(n, )
    XtWX_inv_XtWDX_row_j = XtWX_inv_XtWD_row_j.dot(x_data) # shape(k, )
    beta_all = XtWX_inv_XtW.dot(y_data)

    ### if finalize:
    beta_ij = beta_all[j, 0]
    ### if finalize:

    y_hat_local = x_data.dot(beta_all)

    dy_hat_ij = x_data[i, j] * XtWX_inv_XtWD_row_j.dot(y_data - y_hat_local)[0]

    dtrS_j = - x_data[i, j] * XtWX_inv_XtWDX_row_j.dot(XtWX_inv).dot(x_data[i, :])

    Ri_j = x_data[i, j] * XtWX_inv_XtW[j, :]
    Riij = Ri_j[i]
    Ri_j_sq = np.sqrt(Ri_j.dot(Ri_j.T))

    y_hat_i = Ri_j.dot(y_data)[0] # y_hat_local[i]? TODO

    trS_ij = x_data[i, j] * XtWX_inv_XtW[j, i]

    # if hess:
    #     XtWX_inv_XtWDX_XtWX_inv = XtWX_inv_XtWDX.dot(XtWX_inv)
    #     XtWX_inv_XtWDD = XtWX_inv.dot(x_data.T) * wdd
    #
    #     beta_local_res = XtWX_inv_XtWD.dot(y_data - y_hat_local)
    #     dy_hat_local = x_data.dot(beta_local_res)
    #
    #     temp1 = XtWX_inv_XtWDX[[j], :].dot(XtWX_inv_XtWD).dot(y_data - y_hat_local)
    #     temp2 = XtWX_inv_XtWDD[[j], :].dot(y_data - y_hat_local)
    #     temp3 = XtWX_inv_XtWD[[j], :].dot(y_data - dy_hat_local)
    #
    #     d2y_hat_ij = 2 * x_data[i, [j]] * (temp1 - temp2 - temp3)
    #     d2y_hat_ij = NUM.asarray(d2y_hat_ij).item()
    #
    #     d2trS_j = - x_data[i, [j]] * (-2 * XtWX_inv_XtWDX[[j], :].dot(XtWX_inv_XtWDX_XtWX_inv) + \
    #                                   XtWX_inv_XtWDD[[j], :].dot(x_data).dot(XtWX_inv)).dot(x_data[i, :])
    #     d2trS_j = d2trS_j.item()

    return beta_ij, dy_hat_ij, dtrS_j, d2trS_j, trS_ij, Riij, y_hat_i, Ri_j_sq, d2y_hat_ij

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
    y_hat = NUM.zeros((n, 1), dtype=float)  ### additive (must be init with 0)
    if finalize:
        beta = NUM.zeros((n, k), dtype=float)
        std_beta = NUM.zeros((n, k), dtype=float)
        influence = NUM.zeros((n, 1), dtype=float)
        localR2 = None  # TODO
        RSS = None
        aicc = None
        sigma = None

    trS = 0.  ### additive
    ENPj = NUM.zeros(k, dtype=float)  ### additive
    ################# final output #################

    dy_hat = NUM.zeros((n, k), dtype=float)  # if not finalize: # TODO
    dtrS = NUM.zeros((k, 1), dtype=float)  ### additive

    # if hess:
    #     d2trS = NUM.zeros((k, 1), dtype=float)  ### additive
    #     d2y_hat = NUM.zeros((n, k), dtype=float)  ### additive # TODO can remove, but tricky one

    T_LOOP = 0.
    for i in range(n):
        for j in range(k):
            ################# goes beyond the double-for-loop #################
            # Ri_j = NUM.zeros((n, 1), dtype=float)
            # trS_ij = 0.         # for trS
            # dtrS_j = 0.         # for dtrS
            # dy_hat_ij = 0.      # for dy_hat
            # d2trS_j = 0.        # for d2trS
            # beta_ij = 0.        # for beta
            # # d2_yhat_ij = 0.     # for d2_yhat
            # # y_hat_ij = 0.        # for y_hat
            ################# goes beyond the double-for-loop #################
            # if neighbor_type == "distance":
            #     poly_coef_i = None
            # else:
            #     poly_coef_i = poly_coef[i]
            # i_j_poly_coef_i = [i, j, poly_coef_i]
            # i_j_poly_coef_i = [i, j, poly_coef[[i]]]

            beta_ij, dy_hat_ij, dtrS_j, d2trS_j, trS_ij, Riij, y_hat_i, Ri_j_sq, d2y_hat_ij = \
                lloop(i, j, poly_coef[i, :], hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)
            # TODO if finalize we don;t do parallel, don't have lloop here, slower

            ################# goes beyond the double-for-loop #################
            if finalize:
                beta[i, j] = beta_ij
            dy_hat[i, j] = dy_hat_ij
            dtrS[j] += dtrS_j
            # if hess:
            #     d2trS[j] += d2trS_j
            #     d2y_hat[i, j] = d2y_hat_ij
            trS += trS_ij
            ENPj[j] += Riij
            y_hat[i] += y_hat_i

            if finalize:
                influence[i] += Riij
                std_beta[i, j] = Ri_j_sq * NUM.abs(1. / x_data[i, j])

            ################# goes beyond the double-for-loop #################

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
    daicc = n / RSS * dRSS + 2 * n * (n - 1) / ((n - trS - 2) ** 2) * dtrS
    # if hess:
    #     d2aicc_rss = - n / RSS ** 2 * dRSS.dot(dRSS.T) + n / RSS * d2RSS
    #     d2aicc_trs = 4 * n * (n - 1) / ((n - trS - 2) ** 3) * dtrS.dot(dtrS.T) + 2 * n * (n - 1) / (
    #                 (n - trS - 2) ** 2) * NUM.diag(d2trS)
    #     d2aicc = d2aicc_rss + d2aicc_trs
    ################# Jacobian, Hessian of aicc #################

    if finalize:
        return beta, std_beta, y_hat, trS, ENPj, RSS, aicc, sigma, influence, localR2
    # if test:
        # if hess:
        #     return aicc, daicc, d2aicc, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS
        # else:
        #     return aicc, daicc, RSS, dRSS, trS, dtrS
    return aicc, daicc, d2aicc



### cache off, parallel on
### 5*1000*11 - 69s
### cache on, parallel on
### 1000 -  49s

### cache on, parallel off
### 5*1000*11 - 11s
### 5*5000*11 - 106s

### cache off, parallel off
### 5*1000*11 - 11s
### 5*5000*11 - 104s


if __name__ == '__main__':
    import time
    from kernels import *

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

    njit_time = 0
    lloop(0, 0, poly_coef[0, :], hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data,
          coords_scaled, n, k)

    start = time.time()
    for i in range(n):
        for j in range(k):
            beta_ij, dy_hat_ij, dtrS_j, d2trS_j, trS_ij, Riij, y_hat_i, Ri_j_sq, d2y_hat_ij = \
                lloop(i, j, poly_coef[i, :], hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data,
                      coords_scaled, n, k)
            ################# goes beyond the double-for-loop #################

    end = time.time()
    njit_time += end - start
    print("time ", njit_time)
    # lloop.inspect_types()
    lloop.parallel_diagnostics(level=4)


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
