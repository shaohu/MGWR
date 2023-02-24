import numpy as np
from numpy.linalg import inv as inv
from numba import njit, jit

def opt_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    aicc, daicc, *_ = gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef)
    return aicc, daicc

def hess_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    _, _, d2aicc = gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, True, poly_coef)
    return d2aicc

def opt_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    aicc, daicc, *_ = gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef)
    return aicc, daicc.flatten()

def hess_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef):
    _, _, d2aicc = gradient_mgwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, True, poly_coef)
    return d2aicc

@njit(nogil=True)
def loop_n(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k):
    ## final output
    y_hat = np.zeros((n, 1))

    ### if finalize:
    beta = np.zeros((n, k))
    std_beta = np.zeros((n, k))
    influence = np.zeros((n, 1))
    ENP = 0.
    trS = 0.

    ### used in loop for local fit
    y_hat_local = np.zeros((n, 1))

    dRSS = 0.
    dtrS = 0.
    for i in range(n):
        distances = get_distance(coords_scaled, i)
        ## TODO
        w, wd = kernel_func(distances, banwidth, poly_coef[i, :])

        XtW = x_data.T * w
        XtWX = XtW.dot(x_data)
        XtWX_inv = inv(XtWX)
        XtWX_inv_XtW = XtWX_inv.dot(XtW)
        XtWX_inv_XtWD = XtWX_inv.dot(x_data.T) * wd  ### AiBi in (24)(25)

        trS += x_data[i].dot(XtWX_inv_XtW[:, i])

        beta_all = XtWX_inv_XtW.dot(y_data)

        ### if finalize:
        influence[i] = x_data[i].dot(XtWX_inv_XtW[:, i])
        ENP += influence[i, 0]  # TODO haven't check this
        beta[i] = beta_all.T
        std_beta[i] = np.sqrt(np.diag(XtWX_inv_XtW.dot(XtWX_inv_XtW.T)))  # TODO haven't check this # TODO sqrt

        y_hat_local = x_data.dot(beta_all)
        y_hat[i] = x_data[i].dot(beta_all)
        ## TODO enhance this
        d_yhat_i = x_data[i].dot(XtWX_inv_XtWD).dot(y_data - y_hat_local)[0]  # 1k*kn*n1 = O(kn)
        dRSS += - 2 * d_yhat_i * (y_data[i, 0] - y_hat[i, 0])

        dtrS += - x_data[i].dot(XtWX_inv_XtWD).dot(x_data).dot(XtWX_inv).dot(x_data[i])  # 1k*kn*nk*k1 = O(nk)
    return beta, trS, dRSS, dtrS, ENP, y_hat, influence, std_beta

def gradient_gwr(banwidth, y_data, x_data, coords_scaled, neighbor_type, kernel_func, hess, poly_coef, test=False,
                 finalize=False):
    """
    this function read in data and bandwidth; returns AICc, gradient of AICc and all f(xi) components
    :param banwidth: scalar
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

    beta, trS, dRSS, dtrS, ENP, y_hat, influence, std_beta = \
        loop_n(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)

    ### get fitted value, trace(S), RSS, aicc
    RSS = np.sum((y_data - y_hat) ** 2)
    aicc = n * np.log(RSS / n) + n * np.log(2.0 * np.pi) + n * (trS + n) / (n - trS - 2.0)

    if finalize:
        sigma = np.sqrt(RSS / (n - trS)) # TODO sqrt
        std_beta = std_beta * sigma

    ### Jacobian, Hessian of aicc
    daicc = n / RSS * dRSS + 2 * n * (n - 1) / ((n - trS - 2) ** 2) * dtrS

    localR2 = None
    if finalize:
        return beta, std_beta, y_hat, trS, ENP, RSS, aicc, sigma, influence, localR2

    return aicc, daicc, None

    ###################
    # if hess:
    #     XtWX_inv_XtWDX = np.zeros((k, k), dtype=float)
    #     XtWX_inv_XtWDX_XtWX_inv = np.zeros((k, k), dtype=float)
    #     XtWX_inv_XtWDD = np.zeros((k, n), dtype=float)
    ##################
    ###################
    # if hess:
    #     d2trS = 0.
    #     d2RSS = 0.
    #     d2RSS_all = 0.
    #     d2RSS_diag = 0.
    ###################
        # if hess:
        #     XtWX_inv_XtWDX = XtWX_inv_XtWD.dot(x_data)
        #     XtWX_inv_XtWDX_XtWX_inv = XtWX_inv_XtWDX.dot(XtWX_inv)
        #     XtWX_inv_XtWDD = XtWX_inv.dot(x_data.T) * wdd
        ###################
        # if hess:
        #     d2_yhat_i = x_data[i].dot(-2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWD) + \
        #                               XtWX_inv_XtWDD).dot(y_data - y_hat_local)  # 1k * (kk*kn + kn*n1) = O(k2n)
        #     d2RSS_all += 2 * (d_yhat_i ** 2)
        #     d2RSS_diag += - 2 * d2_yhat_i * (y_data[i] - y_hat[i])
        ###################

        ###################
        # if hess:
        #     d2trS += - x_data[i].dot(- 2 * XtWX_inv_XtWDX.dot(XtWX_inv_XtWDX_XtWX_inv) + \
        #                              XtWX_inv_XtWDD.dot(x_data).dot(XtWX_inv)).dot(x_data[i])  ### ...(28)
        # # 1k * (kk*kk + kn*nk*kk)*k1 = O(k2n)
        ###################
    ###################
    ### Jacobian, Hessian of RSS
    # if hess:
    #     d2RSS = d2RSS_all + d2RSS_diag
    ###################
    ###################
    # d2aicc = None
    # if hess:
    #     d2aicc_rss = - n / RSS ** 2 * dRSS ** 2 + n / RSS * d2RSS  ### ...(18)
    #     d2aicc_trs = 4 * n * (n - 1) / ((n - trS - 2) ** 3) * (dtrS ** 2) + 2 * n * (n - 1) / (
    #             (n - trS - 2) ** 2) * d2trS  ### ...(27)
    #     d2aicc = d2aicc_rss + d2aicc_trs  ### ...(17)
    ###################

    # if test:
    #     if hess:
    #         return aicc, daicc, d2aicc, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS
    #     else:
    #         return aicc, daicc, RSS, dRSS, trS, dtrS
    # if hess:
    #     return aicc, daicc, d2aicc


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

# def conditional_njit(njit, n):
#     if n > 15000:
#         return njit(nogil=True, parallel=True)
#     else:
#         return njit(nogil=True, parallel=False)

@njit(nogil=True, parallel=True)
def matrix_calc(x_data, y_data, w, wd, j):
    """
    return: shape of (n,) (n,) (n,) (n,) (n,)
    """
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

@njit(nogil=True)
def loop_nk(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k):

    y_hat = np.zeros((n, 1))  ### additive
    ### if finalize:
    beta = np.zeros((n, k))
    std_beta = np.zeros((n, k))
    influence = np.zeros((n, 1))
    ### if finalize:
    trS = 0.  ### additive
    ENPj = np.zeros(k)  ### additive
    dy_hat = np.zeros((n, k))
    dtrS = np.zeros((k, 1))  ### additive

    for i in range(n):
        poly_coef_i = poly_coef[i, :]

        for j in range(k):
            banwidth_j = banwidth[j]

            distances = get_distance(coords_scaled, i)

            w, wd = kernel_func(distances, banwidth_j, poly_coef_i)

            XtWX_inv, XtWX_inv_XtW_row_j, XtWX_inv_XtWD_row_j, XtWX_inv_XtWDX_row_j, y_hat_local, beta_ij = matrix_calc(x_data, y_data, w, wd, j)

            dy_hat_ij = x_data[i, j] * XtWX_inv_XtWD_row_j.dot(y_data - y_hat_local)[0]

            dtrS_j = - x_data[i, j] * XtWX_inv_XtWDX_row_j.dot(XtWX_inv).dot(x_data[i, :]) # k1 kk k1

            Ri_j = x_data[i, j] * XtWX_inv_XtW_row_j
            Riij = Ri_j[i]
            Ri_j_sq = np.sqrt(Ri_j.dot(Ri_j.T))

            y_hat_i = Ri_j.dot(y_data)[0]

            trS_ij = x_data[i, j] * XtWX_inv_XtW_row_j[i]

            ### if finalize:
            beta[i, j] = beta_ij

            ### if finalize:
            dy_hat[i, j] = dy_hat_ij
            dtrS[j] += dtrS_j

            trS += trS_ij
            ENPj[j] += Riij
            y_hat[i] += y_hat_i

            ### if finalize:
            influence[i] += Riij
            std_beta[i, j] = Ri_j_sq * np.abs(1. / x_data[i, j])
            # print(std_beta[i, j], np.sqrt(XtWX_inv_XtW_row_j.dot(XtWX_inv_XtW_row_j.T)))
            ### if finalize:

    return beta, dy_hat, dtrS, trS, ENPj, y_hat, influence, std_beta
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
#####################
# XtW = x_data.T * w # *(kn)(n)
# XtWX = XtW.dot(x_data) # (kn)(nk)
# XtWX_inv = inv(XtWX) # (kk inv)
#
# XtWX_inv_Xt = XtWX_inv.dot(x_data.T) # (kk)(kn)
# XtWX_inv_XtW = XtWX_inv.dot(XtW) # (kk)(kn)
# XtWX_inv_XtW_row_j = XtWX_inv_XtW[j, :]
# XtWX_inv_XtWD_row_j = XtWX_inv_Xt[j, :] * wd # (n)(n) = shape(n, )
# XtWX_inv_XtWDX_row_j = XtWX_inv_XtWD_row_j.dot(x_data) # (n)(nk) = shape(k, )
# beta_all = XtWX_inv_XtW.dot(y_data) # (kn)(n1)
# y_hat_local = x_data.dot(beta_all)
# ### if finalize:
# beta_ij = beta_all[j, 0]
# ### if finalize:
#####################
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
# if hess:
#     d2trS[j] += d2trS_j
#     d2y_hat[i, j] = d2y_hat_ij
################# goes beyond the double-for-loop #################

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

    ################# final output #################
    if finalize:
        localR2 = None
        RSS = None
        aicc = None
        sigma = None

    beta, dy_hat, dtrS, trS, ENPj, y_hat, influence, std_beta =  \
        loop_nk(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)

    ################# final output #################
    daicc = np.zeros((k, 1), dtype=float)
    d2aicc = np.zeros((k, k), dtype=float)
    ################# final output #################

    ################# intermediate #################
    dRSS = np.zeros((k, 1), dtype=float)
    ################# intermediate #################

    ################# get output #################
    trS = np.asarray(trS).item()

    RSS = np.sum((y_data - y_hat) ** 2)
    aicc = n * np.log(RSS / n) + n * np.log(2.0 * np.pi) + n * (trS + n) / (n - trS - 2.0)
    sigma = np.sqrt(RSS / (n - trS))

    if finalize:
        std_beta = std_beta * sigma
    ################# get output #################

    ################# Jacobian, Hessian of RSS #################
    dRSS = -2 * dy_hat.T.dot(y_data - y_hat)
    ################# Jacobian, Hessian of RSS #################

    ################# Jacobian, Hessian of aicc #################
    daicc = n / RSS * dRSS + 2 * n * (n - 1) / ((n - trS - 2) ** 2) * dtrS
    ################# Jacobian, Hessian of aicc #################

    if finalize:
        return beta, std_beta, y_hat, trS, ENPj, RSS, aicc, sigma, influence, localR2
    return aicc, daicc, None

    # if hess:
    #     d2trS = NUM.zeros((k, 1), dtype=float)  ### additive
    #     d2y_hat = NUM.zeros((n, k), dtype=float)  ### additive # TODO can remove, but tricky one

    # if hess:
    #     d2RSS_all = NUM.zeros((k, k), dtype=float)
    #     d2RSS = NUM.zeros((k, k), dtype=float)
    #     d2aicc_rss = NUM.zeros((k, k), dtype=float)
    #     d2aicc_trs = NUM.zeros((k, k), dtype=float)

    # if hess:
    #     d2RSS_all = 2 * dy_hat.T.dot(dy_hat)
    #     d2RSS_diag = NUM.sum(d2y_hat * (y_data - y_hat), axis=0)
    #     d2RSS = d2RSS_all + NUM.diag(d2RSS_diag)

    # if hess:
    #     d2aicc_rss = - n / RSS ** 2 * dRSS.dot(dRSS.T) + n / RSS * d2RSS
    #     d2aicc_trs = 4 * n * (n - 1) / ((n - trS - 2) ** 3) * dtrS.dot(dtrS.T) + 2 * n * (n - 1) / (
    #                 (n - trS - 2) ** 2) * NUM.diag(d2trS)
    #     d2aicc = d2aicc_rss + d2aicc_trs

    # if test:
        # if hess:
        #     return aicc, daicc, d2aicc, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS
        # else:
        #     return aicc, daicc, RSS, dRSS, trS, dtrS


########################################################################################################################
## loop_nk, no njit, 27s
## loop_nk, njit,    17s
## loop_nk, nogil=True, parallel=True 73s!
## SVML not helping
## fastmath not helping
## prange not helping (but loops not paralleled anyway)

import time
from kernels import *
if __name__ == '__main__':
        n = 5000
        k = 11
        np.random.seed(1)
        x_data = np.random.random((n, k))
        y_data = np.random.random((n, 1))
        coords_scaled = np.random.random((n, 2))
        poly_coef = np.random.random((n, 6))

        hess = False
        neighbor_type = "neighbors"
        kernel_func = kernel_nb_bisq
        finalize = False

        ###################################################################################################
        ### test loop_nk()
        # banwidth = np.random.randint(0, n, k)
        # njit_time = 0
        # loop_nk(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)
        # start = time.time()
        # loop_nk(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)
        # end = time.time()
        # njit_time += end - start
        # print("time ", njit_time)
        # loop_nk.inspect_types()
        # loop_nk.parallel_diagnostics(level=4)
        ###################################################################################################
        ### test loop_n()
        # banwidth = np.random.random()
        # loop_n(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)
        # loop_n.inspect_types()
        ###################################################################################################


        ###################################################################################################
        ### test how matrix_calc() runtime grows by sample size ~(nk)
        # times = []
        # samples = [1000, 5000, 10000, 20000, 30000, 40000]
        # k = 3
        # for n in samples:
        #     x_data = np.random.random((n, k))
        #     y_data = np.random.random((n, 1))
        #     w = np.random.random(n)
        #     wd = np.random.random(n)
        #     start = time.time()
        #     for i in range(1000):
        #         for j in range(k):
        #             matrix_calc(x_data, y_data, w, wd, j)
        #     end = time.time()
        #     times.append(end-start)
        # samples = np.asarray(samples)
        # times = np.asarray(times)
        # print(samples)
        # print(times)
        # print(times / samples * 1000)
        ########################
        # i=0
        # [1000             5000          10000         20000           30000          40000]
        # [2.31373858      0.01097083     0.0169549     0.03789997      0.05784631     0.07081199]
        # [2.31373858e+00 2.19416618e-03 1.69548988e-03 1.89499855e-03 1.92821026e-03 1.77029967e-03]
        ########################
        # k=11, i for 1000, parallel for matrix_calc
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only take effects when n > 10000, by 10% for 20000 samples, 20% for 40000 samples
        # [1000        5000        10000      20000        30000      40000]
        # [4.8427515   9.87161255 11.86329007 27.52741933 38.23633289 50.74130273]
        # [4.8427515  1.97432251 1.18632901 1.37637097 1.27454443 1.26853257]
        # k=11, i for 1000, no parallel for matrix_calc
        # [1000        5000        10000      20000        30000      40000]
        # [2.98230219  8.09735656 12.00043654 31.45726204 47.09311414 64.42113352]
        # [2.98230219 1.61947131 1.20004365 1.5728631  1.56977047 1.61052834]
        ########################
        # k=3, i for 1000, parallel for matrix_calc
        # [1000        5000      10000      20000      30000      40000]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only take effects when n > 20000, 30000 by 50%
        # [3.50802994 1.0870924  1.80317998 2.95908761 1.33642697 3.68118429]
        # [3.50802994 0.21741848 0.180318   0.14795438 0.04454757 0.09202961]
        # k=3, i for 1000, no parallel for matrix_calc
        # [1000       5000        10000     20000      30000      40000]
        # [2.39293861 0.45928288 0.99434114 2.60853362 2.89326358 5.30432725]
        # [2.39293861 0.09185658 0.09943411 0.13042668 0.09644212 0.13260818]
        ###################################################################################################

        ###################################################################################################
        ### test how loop_nk() runtime grows by sample size
        times = []
        samples = [1000, 5000, 10000, 15000, 20000, 25000]
        k = 10
        hess = False
        neighbor_type = "neighbors"
        kernel_func = kernel_nb_bisq
        finalize = False
        for n in samples:
            x_data = np.random.random((n, k))
            y_data = np.random.random((n, 1))
            coords_scaled = np.random.random((n, 2))
            poly_coef = np.random.random((n, 6))
            banwidth = np.asarray([int(n/2), ] * k)
            start = time.time()
            loop_nk(poly_coef, hess, neighbor_type, kernel_func, banwidth, finalize, x_data, y_data, coords_scaled, n, k)
            end = time.time()
            print(end-start)
            times.append(end-start)
        samples = np.asarray(samples[1:])
        times = np.asarray(times[1:])
        print(samples)
        print(times)
        print(times / samples * 1000)
        print(times / samples / samples * 1000000)
        ######################## ~ n^2.5
        # no parallel
        # [1000         2000           3000       4000       5000            6000          7000       8000        9000         10000]
        # [6.60233974   2.74666095   5.74564481   9.69708586  15.40183711  47.45616865  64.02289248  81.2373929  103.48046207 151.99081755]
        # [6.60233974  1.37333047  1.91521494  2.42427146  3.08036742  7.90936144 9.1461275  10.15467411 11.49782912 15.19908175]
        # [6.60233974 0.68666524 0.63840498 0.60606787 0.61607348 1.31822691 1.30658964 1.26933426 1.27753657 1.51990818]
        ######################## ~ n^2.0+
        # no parallel
        # [5000  10000  15000  20000  25000]
        # [47.18 146.11 401.88 711.40 1072.72]
        # [9.436 14.611 26.792 35.570 42.9091]
        # [1.887 1.4611 1.7861 1.7785 1.71636]
        ########################
        # parallel
        # [5000   10000  15000   20000 25000]
        # [57.417 88.970  444.38
        ###################################################################################################









#######################################################################################################################
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
