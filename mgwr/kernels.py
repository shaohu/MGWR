import numpy as np
from numba import njit

################################### for neighbors #################################
# npoly+1 is the length of smoothing coefficients

global npoly
npoly = 5

### for Poly
global seq1
seq1 = np.arange(npoly+1)[::-1]  # e.g. [5, 4, 3, 2, 1, 0]
### for dPoly
global seq2
seq2 = np.arange(1, npoly+1)[::-1]  # e.g. [5, 4, 3, 2, 1]
global seq3
seq3 = np.arange(npoly+1 - 1)[::-1]  # e.g. [4, 3, 2, 1, 0]

### for ddPoly
# global seq4
# seq4 = np.arange(2, npoly+1)[::-1]  # e.g. [5, 4, 3, 2]
# global seq5
# seq5 = np.arange(1, npoly+1 - 1)[::-1]  # e.g. [4, 3, 2, 1]
# global seq6
# seq6 = np.arange(npoly+1 - 2)[::-1]  # e.g. [3, 2, 1, 0]
################################### for neighbors #################################

@njit(nogil=True)
def get_distance(coords_scaled, i):
    distances = np.sqrt(np.add(coords_scaled[i, 0], -coords_scaled[:, 0].T) ** 2 +
                         np.add(coords_scaled[i, 1], -coords_scaled[:, 1].T) ** 2)
    return distances

@njit(nogil=True)
def check_inf(x):
    """
    replace inf with 0 in array. don't put it back, it would fail numba
    """
    shape = x.shape
    x = x.ravel()
    x[np.isinf(x)] = 0
    x = x.reshape(shape)
    return x

@njit(nogil=True)
def kernel_dist_gaus(distances, bandwidth, polycoef):
    """
    :param distances: a list of distance values against the nearest neighbors (i-th row of distance matrix) of shape (1, n)
    :param bandwidth: a distance value (j-th value of bandwidths) scalar
    :param poly_coef: np.nan
    :return: the weight matrix assigned to the neighbors (w)
            the 1st order derivative of that with respect to distance (wd)
            the 2ed order derivative of that with respect to distance (wde)
            wd is gradient of w, NOT weight*gradient
    gaussian method:
        # w = np.exp(- pow(distances / bandwidth, 2.0) / 2.)
        # d = np.divide(pow(distances, 2.0), pow(bandwidth, 3))
        # wd = w * d
    """

    ### important, avoid overflow in power()
    bandwidth = np.asarray(bandwidth) * 1.00

    k1 = distances / bandwidth
    k2 = np.power(k1, 2.0)
    w = np.exp(- k2 / 2.)
    d = np.divide(k2, bandwidth)
    wd = w * d
    return w, wd

    ##################################
    # wde = None
    # if hess:
    #     e = d - (3. / bandwidth)
    #     wde = wd * e
    ##################################

@njit(nogil=True)
def kernel_dist_bisq(distances, bandwidth, polycoef):
    """
    :param distances: a list of distance values against the nearest neighbors (i-th row of distance matrix) of shape (1, n)
    :param bandwidth: a distance value (j-th value of bandwidths) scalar
    :param poly_coef: np.nan
    :return: the weight matrix assigned to the neighbors (w)
            the 1st order derivative of that with respect to distance (wd)
            the 2ed order derivative of that with respect to distance (wde)
            wd is gradient of w, NOT weight*gradient
    bisquare method:
        # w = pow(1.0 - pow((distances / bandwidth), 2.0), 2.0)
        #
        # temp1 = pow(distances, 2) / pow(bandwidth, 3)
        # temp2 = (1.0 - pow(distances / bandwidth, 2.0))
        # d = 4.0 * np.divide(temp1, temp2)
        # d[np.isinf(d)] = 0 # important, avoid nan when temp2 == 0
        # wd = w * d
        # w[distances > bandwidth] = 0
        # wd[distances > bandwidth] = 0
    """

    ### important, avoid overflow in power()
    bandwidth = np.asarray(bandwidth) * 1.00

    k1 = distances / bandwidth
    k2 = np.power(k1, 2.0)
    k3 = 1.0 - k2

    w = np.power(1.0 - k2, 2.0)
    d = 4.0 * np.divide(np.divide(k2, bandwidth), k3)

    # important, avoid nan output
    d = check_inf(d)

    w = w * (distances <= bandwidth)
    wd = w * d

    return w, wd

    ##################################
    # wd = w * d
    # wde = None
    # if hess:
    #     e = d / 2.0 - 3.0 / bandwidth
    #     wde = wd * e
    #    wde[distances > bandwidth] = 0
    ##################################

@njit(nogil=True)
def kernel_nb_gaus(distances, bandwidth, polycoef):
    """
    :param distances: a list of distance values against the nearest neighbors (i-th row of distance matrix) of shape (1, n)
    :param bandwidth: a value of number of neighbors (j-th value of bandwidths) scalar
    :param polycoef: polynomial coefficent of shape e.g. (1, 6)
    :return: the weight matrix assigned to the neighbors (w)
            the 1st order derivative of that with respect to distance (wdv)
            the 2ed order derivative of that with respect to distance (wd2v)
            wdv is gradient of w, NOT weight*gradient
            this is done by taking derivative of distance with respect to number of neighbors (wdr, wd2r)
    gaussian method:
        # w = np.exp(- pow(distances / Poly, 2.0) / 2.)
        # dr = pow(distances, 2.0) / pow(Poly, 3)
        # wdr = w * dr
        # wdv = wdr * dPoly
    """

    ### important, avoid overflow in power()
    bandwidth = np.asarray(bandwidth) * 1.00

    ### convert number of neighbors banwidth into various bandwidths for location 1, 2, ..., n, scalar
    Poly = np.sum(polycoef * np.power(bandwidth, seq1))

    ### get the gradient of distance with respect to number of neighbors, scalar
    dPoly = np.sum(polycoef[:-1] * (seq2 * np.power(bandwidth, seq3)))

    k1 = distances / Poly
    k2 = np.power(k1, 2.0)
    w = np.exp(- k2 / 2.)
    dr = k2 / Poly
    wdr = w * dr
    wdv = wdr * dPoly

    return w, wdv

    ##################################
    # ddPoly = None
    # if hess:
    #     ddPoly = np.sum(polycoef[:-2] * (seq4 * seq5 * np.power(bandwidth, seq6)), axis=1)
    # wd2r, wd2v = None, None
    # if hess:
    #     er = dr - (3. / Poly)
    #     wd2r = wdr * er
    #     wd2v = dPoly ** 2 * wd2r + wdr * ddPoly
    ##################################

@njit(nogil=True) # , parallel=True
def kernel_nb_bisq(distances, bandwidth, polycoef):
    """
    :param distances: a list of distance values against the nearest neighbors (i-th row of distance matrix) of shape (1, n)
    :param bandwidth: a value of number of neighbors (j-th value of bandwidths) scalar
    :param polycoef: polynomial coefficent of shape e.g. (1, 6)
    :return: the weight matrix assigned to the neighbors (w)
            the 1st order derivative of that with respect to distance (wdv)
            the 2ed order derivative of that with respect to distance (wd2v)
            wdv is gradient of w, NOT weight*gradient
            this is done by taking derivative of distance with respect to number of neighbors (wdr, wd2r)
    bisquare method:
        # w = pow(1.0 - pow((distances / Poly), 2.0), 2.0)
        # temp1 = pow(distances, 2) / pow(Poly, 3)
        # temp2 = (1.0 - pow(distances / Poly, 2.0))
        # dr = 4.0 * np.divide(temp1, temp2)
        # dr[np.isinf(dr)] = 0 # important, avoid nan when temp2 == 0
        #
        # wdr = w * dr
        # wdv = wdr * dPoly
        #
        # w[distances > Poly] = 0
        # wdv[distances > Poly] = 0
    """
    ### important, avoid overflow in power()
    bandwidth = np.asarray(bandwidth) * 1.00

    ### convert number of neighbors banwidth into various bandwidths for location 1, 2, ..., n, scalar
    Poly = np.sum(polycoef * np.power(bandwidth, seq1))

    ### get the gradient of distance with respect to number of neighbors, scalar
    dPoly = np.sum(polycoef[:-1] * (seq2 * np.power(bandwidth, seq3)))

    k1 = distances / Poly
    k2 = np.power(k1, 2.0)
    k3 = 1.0 - k2
    w = np.power(k3, 2.0)
    dr = 4.0 * np.divide(k2 / Poly, k3)

    # important, avoid nan output
    dr = check_inf(dr)

    # bisquare cut-off
    w = w * (distances <= Poly)

    wdr = w * dr
    wdv = wdr * dPoly

    return w, wdv

    ##################################
    # ddPoly = None
    # if hess:
    #     ddPoly = np.sum(polycoef[:-2] * (seq4 * seq5 * np.power(bandwidth, seq6)), axis=1)
    # wd2r, wd2v = None, None
    # if hess:
    #     er = dr / 2. - (3. / Poly)
    #     wd2r = wdr * er
    #     wd2v = dPoly ** 2 * wd2r + wdr * ddPoly
    #     wd2v[distances > Poly] = 0
    ##################################

########################################################################################################################
# with njit, after compile, 48s
# without njit, 380s

if __name__ == '__main__':
    import time

    n = 5000
    k = 11

    coords_scaled = np.random.random((n, 2))
    poly_coef = np.random.random((n, 6))
    bw = np.random.randint(0, n, k)

    ### force compile
    i = 0
    j = 0
    poly_coef_i = poly_coef[i, :]
    distances = np.sqrt(np.add(coords_scaled[i, [0]], -coords_scaled[:, [0]].T) ** 2 +
                        np.add(coords_scaled[i, [1]], -coords_scaled[:, [1]].T) ** 2)
    kernel_nb_bisq(distances, bw[j], poly_coef_i)
    kernel_nb_gaus(distances, bw[j], poly_coef_i)
    kernel_dist_bisq(distances, bw[j], poly_coef_i)
    kernel_dist_gaus(distances, bw[j], poly_coef_i)
    ### force compile


    ### evalute njit
    # njit_time = 0
    # for i in range(n):
    #     distances = np.sqrt(np.add(coords_scaled[i, [0]], -coords_scaled[:, [0]].T) ** 2 +
    #                          np.add(coords_scaled[i, [1]], -coords_scaled[:, [1]].T) ** 2)
    #     start = time.time()
    #     for j in range(k):
    #         poly_coef_i = poly_coef[i, :]
    #
    #         kernel_nb_bisq(distances, bw[j], poly_coef_i)
    #         kernel_nb_gaus(distances, bw[j], poly_coef_i)
    #         kernel_dist_bisq(distances, bw[j], poly_coef_i)
    #         kernel_dist_gaus(distances, bw[j], poly_coef_i)
    #
    #         end = time.time()
    #         njit_time += end - start
    #
    # print("time: ", njit_time, " s")

    ### how runtime grows with sample size ~ (nk) :)
    times = []
    samples = [5000, 10000, 20000, 30000, 40000]
    k = 11
    for n in samples:
        coords_scaled = np.random.random((n, 2))
        poly_coef = np.random.random((n, 6))
        bw = np.ones(k) * int(n/2)

        start = time.time()
        for i in range(5000):
            poly_coef_i = poly_coef[i, :]
            distances = get_distance(coords_scaled, i)
            for j in range(k):
                kernel_nb_bisq(distances, bw[j], poly_coef_i)
        end = time.time()
        times.append(end-start)
    samples = np.asarray(samples)
    times = np.asarray(times)
    print(samples)
    print(times)
    print(times/samples*1000)

    ###
    # no parallel
    # [ 5000 10000 20000 30000 40000]
    # [ 2.76284504  2.70390296  5.19517303  8.61956811 12.93514156]
    # [0.55256901 0.2703903  0.25975865 0.28731894 0.32337854]
    # parallel ### always worse?
    # [ 5000 10000 20000 30000 40000]
    # [22.85840964 19.22782397 18.35590935 11.92254162 13.01936388]
    # [4.57168193 1.9227824  0.91779547 0.39741805 0.3254841 ]
