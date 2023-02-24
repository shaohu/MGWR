"""
outdated
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

class intervalGWR(object):
    """

    """
    def __init__(self, MGWR, nedge=3, file_name = None):
        self.MGWR = MGWR
        self.nedge = nedge

        self.file_name = file_name
        if not file_name:
            self.file_name = "temp_" + str(NUM.random.random()) + ".pdf"

        if self.MGWR.neighbor_type == "distance":
            self.bandwidths = NUM.linspace(self.MGWR.scaled_bounds[0][0], self.MGWR.scaled_bounds[0][1], self.nedge) # TODO heck
            self.raw_bandwidths = NUM.linspace(self.MGWR.bounds[0][0], self.MGWR.bounds[0][1], self.nedge)  # TODO heck

        else:
            self.bandwidths = NUM.unique(NUM.linspace(self.MGWR.bounds[0][0], self.MGWR.bounds[0][1], self.nedge).astype(int))
            self.raw_bandwidths = self.bandwidths
            self.nedge = len(self.bandwidths)

        self.gAICC = NUM.zeros((self.nedge, 1))
        self.gRSS = NUM.zeros((self.nedge, 1))
        self.gtrS = NUM.zeros((self.nedge, 1))

        self.gdAICC = NUM.zeros((self.nedge, 1))
        self.gdRSS = NUM.zeros((self.nedge, 1))
        self.gdtrS = NUM.zeros((self.nedge, 1))

        self.gd2AICC = NUM.zeros((self.nedge, 1))
        self.gd2RSS = NUM.zeros((self.nedge, 1))
        self.gd2trS = NUM.zeros((self.nedge, 1))

    def run(self):
        if self.MGWR.neighbor_type == "distance":
            poly_coef = None
        else:
            poly_coef = self.MGWR.poly_coef

        for i in range(self.nedge):
            bandwidth = [self.bandwidths[i]]
            AICC, dAICC, d2AICC, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, v1, dtrS, d2trS = gradient_gwr(
                bandwidth, self.MGWR.y_data, self.MGWR.x_data, self.MGWR.distance_matrix, self.MGWR.neighbor_type,
                self.MGWR.kernel, hess=True, poly_coef=poly_coef, test=True)

            print("bandwidth {} aicc {}".format(bandwidth, AICC))
            self.gAICC[i] = AICC
            self.gRSS[i] = RSS
            self.gtrS[i] = v1

            self.gdAICC[i] = dAICC
            self.gdRSS[i] = dRSS
            self.gdtrS[i] = dtrS

            self.gd2AICC[i] = d2AICC
            self.gd2RSS[i] = d2RSS
            self.gd2trS[i] = d2trS

        self.getBest()

    def getBest(self):
        from numpy import unravel_index

        self.optAICC = NUM.min(self.gAICC, axis=None)
        self.optBandwidth = []

        for i in unravel_index(self.gAICC.argmin(), self.gAICC.shape):
            self.optBandwidth.append(self.bandwidths[i])

        self.message = "best AICC is " + str(round(self.optAICC, 4)) + "with bandwdith " + str(self.optBandwidth)

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends import backend_pdf as pdf

        fig, axs = plt.subplots(3, 5, figsize=(16, 12), sharey=False)
        plt.subplots_adjust(bottom=0.2, left=.1, right=.9, top=.9, wspace=.35, hspace=.35)
        fig.tight_layout(pad=4.0)

        ps = 3
        cl = "orange"
        ### visualize AICc, Jacobian AICc, Hessian AICc
        i = 0
        axs[i, 0].scatter(self.raw_bandwidths, self.gAICC, s=ps)
        axs[i, 0].title.set_text("AICc")

        axs[i, 1].scatter(self.raw_bandwidths, self.gdAICC, s=ps)
        axs[i, 1].title.set_text("Jacobian AICc")

        axs[i, 2].scatter(self.raw_bandwidths, self.gd2AICC, s=ps)
        axs[i, 2].title.set_text("Hessian AICc")

        ### visualize RSS, Jacobian RSS, Hessian RSS
        i = 1
        axs[i, 0].scatter(self.raw_bandwidths, self.gRSS, s=ps)
        axs[i, 0].title.set_text("RSS")

        axs[i, 1].scatter(self.raw_bandwidths, self.gdRSS, s=ps)
        axs[i, 1].title.set_text("Jacobian RSS")

        axs[i, 2].scatter(self.raw_bandwidths, self.gd2RSS, s=ps)
        axs[i, 2].title.set_text("Hessian RSS")

        ### visualize trace(S), Jacobian trace(S), Hessian trace(S)
        i = 2
        axs[i, 0].scatter(self.raw_bandwidths, self.gtrS, s=ps)
        axs[i, 0].title.set_text("trace(S)")

        axs[i, 1].scatter(self.raw_bandwidths, self.gdtrS, s=ps)
        axs[i, 1].title.set_text("Jacobian trace(S)")

        axs[i, 2].scatter(self.raw_bandwidths, self.gd2trS, s=ps)
        axs[i, 2].title.set_text("Hessian trace(S)")

        for i in range(3):
            for j in range(3):
                if self.MGWR.neighbor_type == "distance":
                    axs[i, j].set_xlabel("bandwidths")
                else:
                    axs[i, j].set_xlabel("number of neighbors")

        ### quality check of Jacobian
        i = 3
        axs[0, i].scatter(NUM.diff(self.gAICC.flatten()), self.gdAICC.flatten()[1:], s=ps, c=cl)
        axs[0, i].set_xlabel("diff(AICc)")
        axs[0, i].set_ylabel("Jacobian AICc")

        axs[1, i].scatter(NUM.diff(self.gRSS.flatten()), self.gdRSS.flatten()[1:], s=ps, c=cl)
        axs[1, i].set_xlabel("diff(RSS)")
        axs[1, i].set_ylabel("Jacobian RSS")

        axs[2, i].scatter(NUM.diff(self.gtrS.flatten()), self.gdtrS.flatten()[1:], s=ps, c=cl)
        axs[2, i].set_xlabel("diff(trace(S))")
        axs[2, i].set_ylabel("Jacobian trace(S)")

        ### quality check of Hessian
        i = 4
        axs[0, i].scatter(NUM.diff(NUM.diff(self.gAICC.flatten())), self.gd2AICC.flatten()[1:-1], s=ps, c=cl)
        axs[0, i].set_xlabel("diff(diff(AICc))")
        axs[0, i].set_ylabel("Hessian AICc")
        axs[1, i].scatter(NUM.diff(NUM.diff(self.gRSS.flatten())), self.gd2RSS.flatten()[1:-1], s=ps, c=cl)
        axs[1, i].set_xlabel("diff(diff(RSS))")
        axs[1, i].set_ylabel("Hessian RSS")

        axs[2, i].scatter(NUM.diff(NUM.diff(self.gtrS.flatten())), self.gd2trS.flatten()[1:-1], s=ps, c=cl)
        axs[2, i].set_xlabel("diff(diff(trace(S)))")
        axs[2, i].set_ylabel("Hessian trace(S)")

        p0 = pdf.PdfPages(self.file_name)
        p0.savefig()
        p0.close()

if __name__ == '__main__':
    # TODO path
    # create a new folder to store test output?
    ### neighbor
    x, y, coords, data_path = get_georgia_data(nx=6)
    md_nb = MGWR(x, y, coords, data_path=data_path, neighbor_type="neighbor", kernel="gaussian", mode="gwr",
                  disp=True, poly=5)
    gwr_nb = intervalGWR(md_nb, nedge=50, file_name= os.path.join(os.path.dirname(__file__), r"test\test_output\test_gwr_neighbor.pdf"))
    gwr_nb.run()
    gwr_nb.plot()

    ### distance
    x, y, coords, data_path = get_georgia_data(nx=6)
    md_dist = MGWR(x, y, coords, data_path=data_path, neighbor_type="distance", kernel="gaussian", mode="gwr",
                  disp=True, poly=5)
    gwr_dist = intervalGWR(md_dist, nedge=50, file_name= os.path.join(os.path.dirname(__file__), r"test\test_output\test_gwr_distance.pdf"))
    gwr_dist.run()
    gwr_dist.plot()

