from optimize import *
from gradients import *
from kernels import *

import scipy.optimize as opt
import pandas as pd
import numpy as np
from scipy.stats import t as tdist
import time


class MGWR(object):
    def __init__(self, x_data, y_data, coords, add_intercept = False,
                 mode="mgwr", neighbor_type = "distance", kernel="gaussian", poly=5,
                 gwr_init=False, aicc_tol = 1e-5, max_iter = 199, disp=False, data_path=None,
                 scale_coordinate=True):
        """
        currently only supports the same range for all explanatory variables
        :param x_data: (n, k)
        :param y_data: (n, 1)
        :param coords: (n, 2)
        :param mode: "mgwr" or "gwr"
        :param neighbor_type: "distance" or "neighbor"
        :param kernel: "gaussian" or "bisquare"
        :param poly: the order of polynomial function of distance-neighbor smoothing
        :param gwr_init: boolean, if True, use initial bandwidth from GWR
        :param aicc_tol: maxmimam aicc change to continue the search
        :param max_iter: maximum number of search iteration
        :param disp: True or False
        :param data_path: string
        """
        self.x_data = x_data
        self.y_data = y_data
        self.coords = coords
        self.n, self.k = x_data.shape
        if add_intercept and not np.all(self.x_data[:, 0] == 1.):
            self.x_data = np.append(np.ones((self.n, 1)), self.x_data, 1)
            self.k = self.k + 1

        if self.n < 30:
            raise("please provide data with at least 30 samples.")

        self.mode = mode
        if neighbor_type in ["distances", "distance"]:
            self.neighbor_type = "distance"
        elif neighbor_type in ["neighbors", "neighbor", "neighbours", "neighbour"]:
            self.neighbor_type = "neighbor"
        else:
            raise("Error: unknown neighbor_type. ")
        self.kernel = kernel
        self.poly = npoly

        self.gwr_init = gwr_init
        if self.mode == "gwr":
            self.gwr_init = False
        self.gwr_time = 0

        self.aicc_tol = aicc_tol
        self.max_iter = max_iter
        self.disp = disp
        self.data_path = data_path
        self.scale_coord(scale_coordinate=scale_coordinate)

        if self.mode == "mgwr":
            self.opt_model = opt_mgwr
        else:
            self.opt_model = opt_gwr

        if self.n > 60:
            # 30 pysal benchmark = 4
            self.lb_neighbors_ = 30
        else:
            self.lb_neighbors_ = 30

        if self.neighbor_type == "neighbor":
            self.dist_nb_smoothing()
        else:
            self.poly_coef = np.zeros((self.n, self.poly+1)) # make it easier for numba signature

        ### hessian function, may not use it
        if self.mode == "gwr":
            self.hess_func = hess_gwr
        else:
            self.hess_func = hess_mgwr

        if self.neighbor_type == "neighbor":
            if self.kernel == "gaussian":
                self.kernel_func = kernel_nb_gaus
            else:
                self.kernel_func = kernel_nb_bisq
        else:
            if self.kernel == "gaussian":
                self.kernel_func = kernel_dist_gaus
            else:
                self.kernel_func = kernel_dist_bisq

        self.set_bound()
        self.set_start()
        self.create_output_table()
        self.set_up()


    def set_up(self):
        self.bandwidth = None
        self.bandwidth_history = []
        self.aicc = None
        self.aicc_history = []
        self.message = None
        self.method = None
        self.runtime = None
        self.niter = None
        self.njev = None
        self.aic = None
        self.df = None
        self.R2 = None
        self.RSS = None
        self.AdjR2 = None
        self.alphaj = None
        self.tj_value = None
        self.sigma = None
        self.MLE = None
        self.y_resid = None
        self.t_beta = None
        self.p_val = None
        self.std_resid = None
        self.cooksd = None

    def call_back(self, pars, *args, **kws):
        self.bandwidth_history.append(pars)

    def scale_coord(self, scale_coordinate=True):
        """
        Scaling coordinates is MUST for best optimization performance
        scale distance controlling the lat/lon ratio, so that
        - the search range of banwdith is stably around (-1, 1)
        - the y-range of polynomial smoothing is stably around (-1, 1)
        """
        if scale_coordinate:
            self.coords_scale_factor = max(self.coords[:, 0].std(axis=0), self.coords[:, 1].std(axis=0))
            self.coords_center = (self.coords[:, 0].mean(axis=0), self.coords[:, 1].mean(axis=0))

            self.coords_scaled = np.zeros(self.coords.shape)
            self.coords_scaled[:, 0] = (self.coords[:, 0] - self.coords_center[0]) / self.coords_scale_factor
            self.coords_scaled[:, 1] = (self.coords[:, 1] - self.coords_center[1]) / self.coords_scale_factor
        else:
            self.coords_scale_factor = np.asarray([1., 1.])
            self.coords_scaled = self.coords.copy()

    def set_bound(self):
        """
        scaled_bounds is scaled the same way coords does
        bounds is not scaled, physical scale
        when "distance", the lower bound is when every location has self.lb_neighbors_
        when "distance", the upper bound is when every location has all locations as neighbors
        """
        self.scaled_bounds = None
        self.bounds = None

        if self.neighbor_type == "distance":
            from scipy.spatial import KDTree
            from scipy import spatial

            ### lower bound
            tree = KDTree(self.coords_scaled)
            dd, _ = tree.query(self.coords_scaled, k=self.lb_neighbors_+1)
            min_dist = max(dd[:, -1])
            del tree

            ### upper bound
            candidates = self.coords_scaled[spatial.ConvexHull(self.coords_scaled).vertices]
            dist_mat = spatial.distance_matrix(candidates, candidates)
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            max_dist = np.sqrt(sum((candidates[i] - candidates[j]) ** 2))
            ### pysal benchmark double max_dist
            max_dist = max_dist * 2.2

            if self.mode == "mgwr":
                self.scaled_bounds = [(min_dist, max_dist), ] * self.k
                self.bounds = [(self.reverse_scaling_distance(min_dist), self.reverse_scaling_distance(max_dist)), ] * self.k
            else:
                self.scaled_bounds = [(min_dist, max_dist)]
                self.bounds = [(self.reverse_scaling_distance(min_dist), self.reverse_scaling_distance(max_dist))]
        else:
            if self.mode == "mgwr":
                self.bounds = [(self.lb_neighbors_, self.n), ] * self.k
                self.scaled_bounds = self.bounds
            else:
                self.bounds = [(self.lb_neighbors_, self.n)]
                self.scaled_bounds = self.bounds

    def set_start(self, aicc_tol=None):
        """
        set the initial bandwidth by GWR (run trust-constr with Jacobian),
        or a simple function of self.n (middle of min-max range)
        """
        if aicc_tol is None:
            aicc_tol = self.aicc_tol

        if self.neighbor_type == "distance":
            if self.mode == "gwr":
                x0 = np.asarray(self.scaled_bounds).mean(axis=None)
            else:
                x0 = np.asarray(self.scaled_bounds).mean(axis=1)

            if self.mode == "mgwr":
                self.x0 = x0
                if self.gwr_init:
                    t0 = time.time()
                    md = opt.minimize(opt_gwr,
                                      x0=x0[0],
                                      args=(self.y_data, self.x_data, self.coords_scaled, self.neighbor_type,
                                            self.kernel_func, False, self.poly_coef),
                                      bounds=[self.scaled_bounds[0]],
                                      jac=True,
                                      method="L-BFGS-B",
                                      tol=aicc_tol, options={"disp": False})
                    self.x0 = np.repeat(md.x[0], self.k)

                    self.gwr_time = time.time() - t0
                    print("initialize banwdith with gwr: ", self.x0)
            else:
                self.x0 = [x0]
                self.gwr_init = False
                self.gwr_time = 0

        else:
            x0 = int((15 + self.n)/2)

            if self.mode == "mgwr":
                self.x0 = [x0, ] * self.k
                if self.gwr_init:
                    t0 = time.time()
                    md = opt.minimize(opt_gwr,
                                      x0=x0,
                                      args=(self.y_data, self.x_data, self.coords_scaled, self.neighbor_type,
                                            self.kernel_func, False, self.poly_coef),
                                      bounds=[self.bounds[0]],
                                      jac=True,
                                      hess=None,
                                      method="trust-constr",
                                      tol=aicc_tol,
                                      options={"disp": False})

                    # round to integer number of neighhors
                    self.x0 = np.repeat(md.x[0], self.k).astype(int)
                    self.gwr_time = time.time() - t0
                    print("initialize banwdith with gwr: ", self.x0, "time ", self.gwr_time)
            else:
                self.x0 = [x0]
                self.gwr_init = False
                self.gwr_time = 0

    def scale_distance(self, x0):
        return np.asarray(x0) / self.coords_scale_factor

    def reverse_scaling_distance(self, x0):
        """
        the opposite process of scale_distance()
        """
        return np.asarray(x0) * self.coords_scale_factor

    def dist_nb_smoothing(self):
        """
        for each location, we smooth the distance over number of neighbors and store the polynomial coefficients
        """
        self.poly_coef = np.zeros((self.n, self.poly+1))

        nbs = np.arange(self.lb_neighbors_, self.n)

        for i in range(self.n):
            distance_i = np.sqrt(np.add(self.coords_scaled[i, [0]], -self.coords_scaled[:, [0]]) ** 2 + \
                                  np.add(self.coords_scaled[i, [1]], -self.coords_scaled[:, [1]]) ** 2).flatten()
            dists = np.sort(distance_i)[self.lb_neighbors_: ]

            polycoef = np.polyfit(nbs, dists, self.poly)
            self.poly_coef[i, :] = polycoef.flatten()

    def create_output_table(self):
        """
        create pandas table to store results
        """
        self.outputResults = pd.DataFrame(
            columns=['data', 'nSample', 'nVariate', 'mode', 'gwr_init', 'neighborType', 'kernel', 'method', 'time',
                     'AICc', 'r2', 'RSS', 'iteration', 'njev', 'bandwidth', 'message', 'bandwidth_scaled', 'bandwidth_history',
                     'aicc_history'])

    def summary_excel(self):
        new_results = {}
        new_results['method'] = self.method
        new_results['time'] = np.round(self.runtime, 2)
        new_results['AICc'] = np.round(self.aicc, 2)
        if self.R2:
            new_results['r2'] = np.round(self.R2, 3)
        if self.RSS:
            new_results['RSS'] = np.round(self.RSS, 3)
        new_results['iteration'] = self.niter
        if self.njev:
            new_results['njev'] = self.njev
        new_results['message'] = self.message

        if self.neighbor_type == "distance":
            new_results['bandwidth'] = np.round(self.bandwidth, 2).tolist().copy()
            new_results['bandwidth_scaled'] = np.round(self.bandwidth_scaled, 2).tolist().copy()
            new_results['bandwidth_history'] = np.round(self.bandwidth_history, 2).tolist().copy()
        else:
            new_results['bandwidth'] = np.asarray(self.bandwidth).astype(int).tolist().copy()
            new_results['bandwidth_scaled'] = None
            new_results['bandwidth_history'] = np.asarray(self.bandwidth_history).astype(int).tolist().copy()

        new_results['aicc_history'] = np.round(self.aicc_history, 2).tolist().copy()

        if new_results:
            new_results['data'] = self.data_path
            new_results['nSample'] = self.n
            new_results['nVariate'] = self.k
            new_results['mode'] = self.mode
            new_results['neighborType'] = self.neighbor_type
            new_results['kernel'] = self.kernel
            new_results['gwr_init'] = self.gwr_init

            self.outputResults = self.outputResults.append(new_results, ignore_index=True)

    def scipy_optimize(self, hess=False, method="trust-constr", maxiter=None, result_file=None, recover_history=True):
        """
        call scipy.optimize()
        ref: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.show_options.html
        ref: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        :param hess: boolean, use Hessian matrix or not
        :param aicc_tol: maxmimam aicc change to continue the search
        :param method: "trust-constr", "L-BFGS-B", "TNC"
        :param maxiter: maximum number of search iterations
        :param recover_history: determines to use a stupid way to recover aicc or not
        :return: none
        """

        self.set_up()
        self.method = "{} (hess {})".format(method, hess)
        print(" ======================== " + self.method + " ========================  ")

        if maxiter is None:
            maxiter = self.max_iter

        jac = True

        options = {"disp": self.disp, "maxiter": maxiter}

        if self.disp:
            if method == "trust-constr":
                options["verbose"] = 2

        ######################################### control ########################################
        ### if changing this section, be cautious and do multiple tests

        ### if use 0.01 in "ftol", "L-BFGS-B" would stop too early
        if method == "L-BFGS-B":
            options["ftol"] = 1e-5

        ### "gtol" is MUST-CONTROL, decreasing it would make "TNC" and "trust-constr" slower, way more iterations
        if method in ["trust-constr", "TNC"]:
            if self.neighbor_type == "neighbor":
                options["gtol"] = 1e-2
            else:
                options["gtol"] = 1e-4

        ### increase "optimality" would make "trust-constr" stop earlier but worse aicc
        # options["optimality"] = 0.1
        # if self.neighbor_type == "neighbor" and method == "trust-constr":
        #     options["xtol"] = 0.5 # "trust-constr" even slower...
        ######################################### control ########################################

        t0 = time.time()
        if hess:
            md = opt.minimize(self.opt_model, x0=np.asarray(self.x0),
                              args=(self.y_data, self.x_data, self.coords_scaled, self.neighbor_type,
                                    self.kernel_func, hess, self.poly_coef),
                              bounds=self.scaled_bounds, jac=jac, hess=self.hess_func, method=method,
                              options=options, callback=self.call_back)
        else:
            md = opt.minimize(self.opt_model, x0=np.asarray(self.x0),
                              args=(self.y_data, self.x_data, self.coords_scaled, self.neighbor_type,
                                    self.kernel_func, hess, self.poly_coef),
                              bounds=self.scaled_bounds, jac=jac, method=method,
                              options=options, callback=self.call_back)

        self.runtime = time.time() - t0 + self.gwr_time
        self.niter = md.nit
        self.njev = md.njev if hasattr(md, 'njev') else md.nit
        self.aicc = md.fun
        self.message = md.message


        if self.neighbor_type == "distance":
            self.bandwidth_scaled = md.x
            self.bandwidth = self.reverse_scaling_distance(md.x)
        else:
            self.bandwidth = np.asarray(md.x).astype(int)

        ### reproduce the aicc history
        if recover_history:
            temp_bandwidth_history = self.bandwidth_history.copy()
            self.bandwidth_history = []

            if self.neighbor_type == "distance":
                for bw in temp_bandwidth_history:
                    self.bandwidth_history.append(bw)
                    aicc, *_ = self.opt_model(np.asarray(bw), self.y_data, self.x_data, self.coords_scaled,
                                             self.neighbor_type, self.kernel_func, False, self.poly_coef)
                    self.aicc_history.append(aicc)
            else:
                for bw in temp_bandwidth_history:
                    self.bandwidth_history.append(np.asarray(bw).astype(int).tolist())
                    aicc, *_ = self.opt_model(np.asarray(bw).astype(int), self.y_data, self.x_data, self.coords_scaled,
                                             self.neighbor_type, self.kernel_func, False, self.poly_coef)
                    self.aicc_history.append(aicc)
        ### reproduce the aicc history
        self.result_store(result_file=result_file)
        self.summary_excel()

    def newton(self, hess=True, learning_rate=1, aicc_tol = 1e-5, min_iter = 5, maxiter = None, result_file=None):

        self.set_up()
        self.method= "Newton (hess {} , alpha {})".format(hess, learning_rate)
        print(" ======================== {} ========================  ".format(self.method))

        if maxiter is None:
            maxiter = self.max_iter

        t0 = time.time()
        bws, self.aicc, self.niter, self.message, self.bandwidth_history, self.aicc_history = \
            minimize_mgwr(self.x0, self.scaled_bounds, self.y_data, self.x_data, self.coords_scaled,
                         self.neighbor_type, self.kernel_func, self.mode, poly_coef=self.poly_coef,
                         disp=self.disp, aicc_tol=aicc_tol, learning_rate=learning_rate, hess=hess,
                          max_iter=maxiter)
        self.runtime = time.time() - t0 + self.gwr_time

        if self.neighbor_type == "distance":
            self.bandwidth_scaled = bws
            self.bandwidth = self.reverse_scaling_distance(bws)
        else:
            self.bandwidth = bws

        self.result_store(result_file=result_file)
        self.summary_excel()

    def pysal_exec(self, result_file=None):
        """
        call mgwr library with golden search by ASU
        """

        self.set_up()

        self.method = "backfitting".format(self.neighbor_type, self.kernel)
        print(" ======================== {} ========================  ".format(self.method))
        from mgwr.gwr import MGWR as PySalMGWR
        from mgwr.gwr import GWR as PySalGWR
        from mgwr.sel_bw import Sel_BW

        multi = self.mode == "mgwr"
        fixed = self.neighbor_type == "distance"

        if self.neighbor_type == "distance":
            coords = self.coords_scaled
        else:
            coords = self.coords

        if self.mode == "mgwr":
            if self.neighbor_type == "distance":
                temp_bw = list(zip(*self.scaled_bounds))
            else:
                temp_bw = list(zip(*self.bounds))
            multi_bw_min = list(temp_bw[0])
            multi_bw_max = list(temp_bw[1])
        else:
            bw_min = self.scaled_bounds[0][0]
            bw_max = self.scaled_bounds[0][1]


        t0 = time.time()
        if self.mode == "mgwr":
            mgwr_selector = Sel_BW(coords, self.y_data, self.x_data, multi=multi, constant=False, kernel=self.kernel, fixed=fixed)

            bws = mgwr_selector.search(search_method='golden_section', multi_bw_min=multi_bw_min, multi_bw_max=multi_bw_max, verbose=self.disp)

            results = PySalMGWR(coords, self.y_data, self.x_data, mgwr_selector, constant=False, kernel=self.kernel, fixed=fixed).fit()

            print(results.summary())

            self.bandwidth_history = np.asarray([[i[-1][0]] for i in mgwr_selector.sel_hist]).reshape((-1, self.k)).tolist()

        else:
            gwr_selector = Sel_BW(coords, self.y_data, self.x_data, multi=multi, constant=False, kernel=self.kernel, fixed=fixed)

            bws = gwr_selector.search(search_method='golden_section', bw_min=bw_min, bw_max=bw_max, verbose=self.disp)

            results = PySalGWR(coords, self.y_data, self.x_data, bws, constant=False, kernel=self.kernel, fixed=fixed).fit()

            print(results.summary())

            self.bandwidth_history = [[round(i[0], 4)] for i in gwr_selector.sel_hist]

        self.runtime = time.time() - t0
        self.niter = len(self.bandwidth_history)
        self.aicc = results.aicc
        if self.mode == "mgwr":
            if self.neighbor_type == "distance":
                self.bandwidth_scaled = bws.tolist()
                self.bandwidth = [self.reverse_scaling_distance(bws)]
            else:
                self.bandwidth = bws.tolist()
        else:
            if self.neighbor_type == "distance":
                self.bandwidth_scaled = [bws]
                self.bandwidth = [self.reverse_scaling_distance(bws)]
            else:
                self.bandwidth = [bws]

        self.result_store(result_file=result_file)
        self.summary_excel()

    def result_store(self, print_output = True, result_file = None):
        """
        print detailed output and create csv output including beta, std_beta etc
        """


        if self.neighbor_type == "distance":
            bw = self.bandwidth_scaled
        else:
            bw = self.bandwidth

        ### family-wise error rate
        eta = 0.05

        ########################### Dignostics ###########################
        if self.mode == "mgwr":
            self.beta, self.std_beta, self.y_hat, self.v1, self.ENPj, self.RSS, self.aicc, self.sigma, self.influence, self.localR2 = \
                gradient_mgwr(bw, self.y_data, self.x_data, self.coords_scaled, self.neighbor_type, self.kernel_func,
                              hess=False, poly_coef=self.poly_coef, finalize=True)
        else:
            self.beta, self.std_beta, self.y_hat, self.v1, self.ENPj, self.RSS, self.aicc, self.sigma, self.influence, self.localR2 = \
                gradient_gwr(bw[0], self.y_data, self.x_data, self.coords_scaled, self.neighbor_type, self.kernel_func,
                              hess=False, poly_coef=self.poly_coef, finalize=True)

        self.df = self.n - self.v1
        self.R2 = 1 - self.RSS / np.sum(self.y_data ** 2)
        self.AdjR2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.v1 - 1)
        self.alphaj = eta / self.ENPj
        self.tj_value = tdist.ppf(1 - np.abs(self.alphaj) / 2, self.n - 1)
        self.sigma = np.sqrt(self.RSS / (self.n - self.v1))
        self.MLE = - np.log(self.RSS) * self.n / 2. - (1 + np.log(np.pi / (self.n / 2))) * self.n / 2
        self.aic = -2.0 * self.MLE + 2.0 * (self.v1 + 1)
        self.aicc = -2.0 * self.MLE + 2.0 * self.n * (self.v1 + 1.0) / (self.n - self.v1 - 2.0)

        summary = ''
        summary += "%s\n" % ('MGWR bandwidths')
        summary += '-' * 80 + '\n'
        summary += "%-20s %14s %10s %16s %16s\n" % ('Variable', 'Bandwidth', 'ENP_j', 'Adj t-val(95%)', 'DoD_j')
        for j in range(self.k):
            summary += "%-20s %14.3f %10.3f %16.3f %16.3f\n" % (
                "X" + str(j), self.bandwidth[j], self.ENPj[j],  self.tj_value[j], self.alphaj[j])

        summary += "\n%s\n" % ('Diagnostic Information')
        summary += '-' * 80 + '\n'
        summary += "%-67s %12.3f\n" % ('Residual sum of squares:', self.RSS)
        summary += "%-67s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.v1)
        summary += "%-67s %12.3f\n" % ('Degree of freedom (n - trace(S)):', self.df)
        summary += "%-67s %12.3f\n" % ('Sigma estimate:', self.sigma)
        summary += "%-67s %12.3f\n" % ('Log-likelihood:', self.MLE)
        # summary += "%-67s %12.3f\n" % ('Degree of Dependency (DoD):', )
        summary += "%-67s %12.3f\n" % ('AIC:', self.aic)
        summary += "%-67s %12.3f\n" % ('AICc:', self.aicc)
        # summary += "%-67s %12.3f\n" % ('BIC:', '')
        summary += "%-67s %12.3f\n" % ('R2:', self.R2)
        summary += "%-67s %12.3f\n" % ('Adj. R2:', self.AdjR2)

        summary += "\n%s\n" % ('Summary Statistics For MGWR Parameter Estimates')
        summary += '-' * 80 + '\n'
        summary += "%-25s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD', 'Min', 'Median', 'Max')
        summary += "%-25s %10s %10s %10s %10s %10s\n" % ('-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
        for j in range(self.k):
            summary += "%-25s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
                "X" + str(j), np.mean(self.beta[:, j]), np.std(self.beta[:, j]), np.min(self.beta[:, j]),
                np.median(self.beta[:, j]), np.max(self.beta[:, j]))

        summary += '=' * 80 + '\n'

        if print_output:
            print(summary)

        ########################### Local Statistics ###########################
        ### get fitted value and residuals
        self.y_resid = self.y_data - self.y_hat

        ### statistical test of local coefficients
        self.t_beta = self.beta / self.std_beta
        self.p_val =  np.abs(self.t_beta) >= self.tj_value

        ### get influence, std of residuals, cook's distance
        ### influ is diagonal of S matrix, ref https://buildmedia.readthedocs.org/media/pdf/mgwr/latest/mgwr.pdf
        ### ref https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/20/lecture-20.pdf
        self.std_resid = self.y_resid / self.sigma / np.sqrt(1 - self.influence)

        self.cooksd = self.std_resid ** 2 * self.influence / (1 - self.influence) / self.v1

        df = pd.concat([pd.DataFrame(self.beta), pd.DataFrame(self.std_beta), pd.DataFrame(self.t_beta), pd.DataFrame(self.p_val)], axis=1)
        df.columns = ['coef_' + str(ii) for ii in np.arange(self.k)] + \
                       ['std_' + str(ii) for ii in np.arange(self.k)] + \
                       ['t_value_' + str(ii) for ii in np.arange(self.k)] + \
                       ['p_val_' + str(ii) for ii in np.arange(self.k)]

        df['y_fitted'] = self.y_hat
        df['y_resid'] = self.y_resid
        df['std_resid'] = self.std_resid
        df['influence'] = self.influence
        df['cooksd'] = self.cooksd
        # df['cnd_number'] = cnd_number
        # df['localR2'] = localR2

        if result_file:
            df.to_csv(result_file)
            print(result_file, result_file.split(".")[0] + '.txt')
            with open(result_file.split(".")[0] + '.txt', 'w') as f:
                f.write(summary)