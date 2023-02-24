from gradients import *
import numpy as np
from numpy.linalg import inv as inv

def update_bandwidth(x0, bounds, AICC, dAICC, d2AICC, neighbor_type, learning_rate = 1, hess=False):
    """
    this function read in the current bandwidth, AICc, and gradient of AICc; return the bet bandwidth option.
    :param x0: current bandwidths
    :param bounds: range of bandwidths
    :param dAICC: Jacobian matrix of AICC with respect to bandwidth
    :param d2AICC: Hessian matrix of AICC with respect to bandwidth
    :param learning_rate: learning rate
    :param hess: use second order of derivatives,
                    We can remove this and stick with "hessian" after testing.
    :return: next bandwidth option
    """
    x0 = np.asarray(x0)
    dAICC = np.asarray(dAICC).flatten()

    if len(x0) == 1 and hess: # GWR
        d2AICC = np.abs(d2AICC) # so that it finds minima only, otherwise it can be minima or maxmima
        x0_next = x0 - 1. / d2AICC * learning_rate
    elif hess: # MGWR with hess
        invD2 = inv(d2AICC)
        x0_next = x0 - invD2.dot(dAICC).flatten() * learning_rate

        ## check whether inv(d2) positive definite or not
        # print("diag of d2.inv(d2)", np.diagonal(d2AICC.dot(invD2)))
    else: # MGWR without hess
        x0_next = x0 - dAICC * learning_rate

    x0_next = check_bound(x0_next, bounds)
    if neighbor_type != "distance":
        x0_next = x0_next.astype(np.int)
    return x0_next

def check_bound(x0, bounds):
    ### put the bandwidth back to range if they goes beyond the specified range
    for i in range(len(x0)):
        if x0[i] < bounds[i][0]:
            x0[i] = bounds[i][0]
        if x0[i] > bounds[i][1]:
            x0[i] = bounds[i][1]
    return x0

def minimize_mgwr(x0, bounds, y_data, x_data, coords_scaled, neighbor_type, kernel_func, mode, poly_coef=None,
                 aicc_tol=1e-5, xtol = 1e-5, max_iter = 199, min_iter = 5,
                 learning_rate = 1, hess = False, disp = False):
    """
    this function also works for GWR model?
    we dont use SOC as Fotheringham (2017) partly because that requres storing all previous f(x), n*n memory
    :param x0: initial bandwidths
    :param bounds: search range of bandwidths
    :param y_data: y of shape (n, 1)
    :param x_data: x of shape (n, k)
    :param SOCf_tol: tolerance of SOC-f representing the overall change in f(xij), SOC-f as defined by MGWR Fotheringham (2017)
    :param aicc_tol: tolerance of change in AICC
    :param max_iter: maximum number of iterations
    :param learning_rate: learning rate
    :param hess: boolean, use Hessian matrix information or not
    :return: optimal bandwidth, optimal AICC, number of iterations, stopping condition ...
    """

    if mode == "mgwr":
        gradient_func = gradient_mgwr
    else:
        gradient_func = gradient_gwr

    n = x_data.shape[0]

    ### initial run
    if neighbor_type == "distance":
        aicc_cur, daicc_cur, d2aicc_cur, *_ = gradient_func(x0, y_data, x_data, coords_scaled, neighbor_type,
                                                              kernel_func, hess, poly_coef)

    else:
        aicc_cur, daicc_cur, d2aicc_cur, *_ = gradient_func(x0, y_data, x_data, coords_scaled, neighbor_type,
                                                              kernel_func, hess, poly_coef)
    x_cur = x0

    ### message of stopping condition
    stop_message = ""

    ### help controling the loop
    niter = 0
    stop = False

    x_path = [x_cur]
    aicc_path = [aicc_cur]

    while not stop:
        niter += 1

        ### get next bandwidth option
        x_next = update_bandwidth(x_cur, bounds, aicc_cur, daicc_cur, d2aicc_cur, neighbor_type,
                                  learning_rate=learning_rate, hess=hess)

        ### check whether inv(d2) is problematic
        # print("condition number ", np.linalg.cond(d2aicc_cur))

        ### run next MGWR option
        if neighbor_type == "distance":
            aicc_next, daicc_next, d2aicc_next, *_ = gradient_func(x_next, y_data, x_data, coords_scaled,
                                                                      neighbor_type, kernel_func, hess, poly_coef)
        else:
            aicc_next, daicc_next, d2aicc_next, *_ = gradient_func(x_next, y_data, x_data, coords_scaled,
                                                                      neighbor_type, kernel_func, hess, poly_coef)

        ### compare the last AICc and the new AICc
        aicc_delta = aicc_next - aicc_cur

        if disp:
            print("bandwidth {} aicc {}".format(x_next, aicc_next))

        ### important end conditions
        if niter > max_iter:
            stop_message += "\t end condition: niter > " + str(max_iter)
            stop = True

        if abs(aicc_delta) < aicc_tol:
            stop_message += "\t end condition: aicc_delta < " + str(aicc_tol)
            stop = True

        ### to avoid lingering results
        if check_search_path(x_path, x_next, xtol=xtol):
            stop_message += "\t end condition: xtol < " + str(xtol)
            stop = True

        if niter < min_iter:
            stop_message = ""
            stop = False
        ### important end conditions

        x_cur = x_next
        aicc_cur = aicc_next
        daicc_cur = daicc_next
        d2aicc_cur = d2aicc_next
        x_path.append(x_next)
        aicc_path.append(aicc_next)

        ### pick the best one from all iterations
        x_best, aicc_best = optimal_search(x_path, aicc_path)
        ### use the latest one as the best
        # x_best, aicc_best = x_cur, aicc_cur

    return x_best, aicc_best, niter, stop_message, x_path, aicc_path

def check_search_path(x_path, x_cur, xtol):
    """
    check if the current bandwidths are very close to any previous one
    check_search_path([[1,2,3], [1,2,3.001]], [1,2,3.004])
    check_search_path([[1,2,3], [1,2,3.001]], [1,2,3.0001])
    """
    stop = False
    for x_i in x_path:
        if np.all(np.abs(np.asarray(x_i) - np.asarray(x_cur)) < xtol):
            stop = True
    return stop

def optimal_search(x_path, aicc_path):
    """
    find out the lowest aicc along the searching path.
    """
    aicc_path = np.asarray(aicc_path)
    idx = np.argmin(aicc_path)
    return (x_path[idx], aicc_path[idx])
