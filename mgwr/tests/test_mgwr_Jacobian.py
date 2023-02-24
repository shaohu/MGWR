"""
outdated
"""

import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

class gridMGWR(object):
    def __init__(self, MGWR, nedge=3, txt_name=None, file_name=None):

        if MGWR.k != 2:
            raise RuntimeError('gridMGWR works for 2-dim inputs only.')

        self.MGWR = MGWR
        self.nedge = nedge

        self.txt_name = txt_name
        if not txt_name:
            self.txt_name = "temp_" + str(NUM.random.random()) + ".txt"

        self.file_name = file_name
        if not file_name:
            self.file_name = "temp_" + str(NUM.random.random()) + ".pdf"

        if self.MGWR.neighbor_type == "distance":
            self.bandwidths = NUM.linspace(self.MGWR.scaled_bounds[0][0], self.MGWR.scaled_bounds[0][1], self.nedge)
            self.raw_bandwidths = NUM.linspace(self.MGWR.bounds[0][0], self.MGWR.bounds[0][1], self.nedge)
            self.poly_coef = None
        else:
            self.bandwidths = NUM.linspace(self.MGWR.bounds[0][0], self.MGWR.bounds[0][1], self.nedge).astype(int)
            self.raw_bandwidths = NUM.linspace(self.MGWR.bounds[0][0], self.MGWR.bounds[0][1], self.nedge).astype(int)
            self.poly_coef = self.MGWR.poly_coef

        self.gAICC = NUM.zeros((self.nedge, self.nedge))
        self.gdAICC = NUM.zeros((self.nedge, self.nedge, 2))

        self.gRSS = NUM.zeros((self.nedge, self.nedge))
        self.gdRSS = NUM.zeros((self.nedge, self.nedge, 2))

        self.gtrS = NUM.zeros((self.nedge, self.nedge))
        self.gdtrS = NUM.zeros((self.nedge, self.nedge, 2))

        # if self.MGWR.hess:
        #     self.gd2AICC = NUM.zeros((self.nedge, self.nedge, 4))
        #     self.gd2RSS = NUM.zeros((self.nedge, self.nedge, 4))
        #     self.gd2trS = NUM.zeros((self.nedge, self.nedge, 4))

    def getVV(self, a):
        a_max = NUM.max(a)
        a_min = NUM.min(a)

        if a_max > 0:
            if a_min < 0:
                return "Spectral", -NUM.max(NUM.abs(a)), NUM.max(NUM.abs(a))
            return "YlOrRd", a_min, a_max
        return "YlGnBu", a_min, a_max

    def run(self):
        for i in range(self.nedge):
            for j in range(self.nedge):
                bandwidth = [self.bandwidths[i], self.bandwidths[j]]

                AICC, dAICC, RSS, dRSS, trS, dtrS = gradient_mgwr(
                    bandwidth, self.MGWR.y_data, self.MGWR.x_data, self.MGWR.distance_matrix, self.MGWR.neighbor_type,
                    self.MGWR.kernel, hess=False, poly_coef=self.poly_coef, test=True)

                    # AICC, dAICC, d2AICC, d2aicc_rss, d2aicc_trs, RSS, dRSS, d2RSS, trS, dtrS, d2trS = onestepMGWR(
                    #     bandwidth, self.MGWR.y_data, self.MGWR.x_data, self.MGWR.distance_matrix, self.MGWR.neighbor_type,
                    #     self.MGWR.kernel, False, poly_coef=self.poly_coef, mode="testing")

                print("bandwidth {} aicc {}".format(bandwidth, AICC))

                self.gAICC[i, j] = AICC
                self.gdAICC[i, j] = dAICC.flatten()

                self.gRSS[i, j] = RSS
                self.gdRSS[i, j] = dRSS.flatten()

                self.gtrS[i, j] = trS
                self.gdtrS[i, j] = dtrS.flatten()

                # if self.MGWR.hess:
                #     self.gd2AICC[i, j] = d2AICC
                #     self.gd2RSS[i, j] = d2RSS
                #     self.gd2trS[i, j] = d2trS

        self.getBest()

    def storeResults(self):
        f = open(self.txt_name, 'w')
        f.write(u' bandwidth \n')
        f.write(u'#' + '\t'.join(str(e) for e in self.bandwidths) + '\n')
        f.write(u' AICC \n')
        for irow in self.gAICC:
            f.write(u'#' + '\t'.join(str(e) for e in irow) + '\n')
        f.write(u' dAICC \n')
        for irow in self.gdAICC:
            f.write(u'#' + '\t'.join(str(e) for e in irow) + '\n')
        # f.write(u' d2AICC \n')
        # for irow in self.gd2AICC:
        #     f.write(u'#' + '\t'.join(str(e) for e in irow) + '\n')
        f.close()

    def getBest(self):
        from numpy import unravel_index

        self.optAICC = NUM.min(self.gAICC, axis=None)
        self.optBandwidth = []

        for i in unravel_index(self.gAICC.argmin(), self.gAICC.shape):
            self.optBandwidth.append(self.bandwidths[i])

        self.message = "best AICc is " + str(round(self.optAICC, 4)) + " with bandwdith " + str(self.optBandwidth)

    def plot(self):
        """
        r1 and r2 are two variables
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.backends import backend_pdf as pdf

        dAICC_r1 = (self.gAICC.T[:, 1:] - self.gAICC.T[:, :-1]).T
        dAICC_r1_r1 = (dAICC_r1.T[:, 1:] - dAICC_r1.T[:, :-1]).T
        dAICC_r1_r2 = (dAICC_r1[:, 1:] - dAICC_r1[:, :-1])

        dAICC_r2 = (self.gAICC[:, 1:] - self.gAICC[:, :-1])
        dAICC_r2_r1 = (dAICC_r2.T[:, 1:] - dAICC_r2.T[:, :-1]).T
        dAICC_r2_r2 = (dAICC_r2[:, 1:] - dAICC_r2[:, :-1])

        dRSS_r1 = (self.gRSS.T[:, 1:] - self.gRSS.T[:, :-1]).T
        dRSS_r1_r1 = (dRSS_r1.T[:, 1:] - dRSS_r1.T[:, :-1]).T
        dRSS_r1_r2 = (dRSS_r1[:, 1:] - dRSS_r1[:, :-1])

        dRSS_r2 = (self.gRSS[:, 1:] - self.gRSS[:, :-1])
        dRSS_r2_r1 = (dRSS_r2.T[:, 1:] - dRSS_r2.T[:, :-1]).T
        dRSS_r2_r2 = (dRSS_r2[:, 1:] - dRSS_r2[:, :-1])

        dtrS_r1 = (self.gtrS.T[:, 1:] - self.gtrS.T[:, :-1]).T
        dtrS_r1_r1 = (dtrS_r1.T[:, 1:] - dtrS_r1.T[:, :-1]).T
        dtrS_r1_r2 = (dtrS_r1[:, 1:] - dtrS_r1[:, :-1])

        dtrS_r2 = (self.gtrS[:, 1:] - self.gtrS[:, :-1])
        dtrS_r2_r1 = (dtrS_r2.T[:, 1:] - dtrS_r2.T[:, :-1]).T
        dtrS_r2_r2 = (dtrS_r2[:, 1:] - dtrS_r2[:, :-1])

        i, j, z = 3, 5, 7
        vis_dict = {
            "AICc": [0, 0, self.gAICC],
            "diff(AICc) wrt r1": [0, 1, dAICC_r1],
            "diff(AICc) wrt r2": [0, 4, dAICC_r2],
            "Jacobian AICc wrt r1": [0, 2, self.gdAICC[:, :, 0]],
            "Jacobian AICc wrt r2": [0, 5, self.gdAICC[:, :, 1]],
            "diff(AICc) vs Jacobian AICc wrt r1": [0, 3, dAICC_r1, self.gdAICC[1:, :, 0], "diff(AICc) wrt r1", "Jacobian AICc wrt r1"],
            "diff(AICc) vs Jacobian AICc wrt r2": [0, 6, dAICC_r2, self.gdAICC[:, 1:, 1], "diff(AICc) wrt r2", "Jacobian AICc wrt r2"],

            "RSS": [1, 0, self.gRSS],
            "diff(RSS) wrt r1": [1, 1, dRSS_r1],
            "diff(RSS) wrt r2": [1, 4, dRSS_r2],
            "Jacobian RSS wrt r1": [1, 2, self.gdRSS[:, :, 0]],
            "Jacobian RSS wrt r2": [1, 5, self.gdRSS[:, :, 1]],
            "diff(RSS) vs Jacobian RSS wrt r1": [1, 3, dRSS_r1, self.gdRSS[1:, :, 0], "diff(RSS) wrt r1", "Jacobian RSS wrt r1"],
            "diff(RSS) vs Jacobian RSS wrt r2": [1, 6, dRSS_r2, self.gdRSS[:, 1:, 1], "diff(RSS) wrt r2", "Jacobian RSS wrt r2"],

            "trace(S)": [2, 0, self.gtrS],
            "diff(trace(S)) wrt r1": [2, 1, dtrS_r1],
            "diff(trace(S)) wrt r2": [2, 4, dtrS_r2],
            "Jacobian trace(S) wrt r1": [2, 2, self.gdtrS[:, :, 0]],
            "Jacobian trace(S) wrt r2": [2, 5, self.gdtrS[:, :, 1]],
            "diff(trace(S)) vs Jacobian trace(S) wrt r1": [2, 3, dtrS_r1, self.gdtrS[1:, :, 0], "diff(trace(S)) wrt r1", "Jacobian trace(S) wrt r1"],
            "diff(trace(S)) vs Jacobian trace(S) wrt r2": [2, 6, dtrS_r2, self.gdtrS[:, 1:, 1], "diff(trace(S)) wrt r2", "Jacobian trace(S) wrt r2"],

            # "dAICC_r1": [i, 0, dAICC_r1],
            # "dAICC_r1_r1": [i, 1, dAICC_r1_r1],
            # "dAICC_r1_r2": [i, 2, dAICC_r1_r2],
            # "output_d2aicc[:,:,0,0]": [i, 3, self.gd2AICC[:, :, 0, 0]],
            # "output_d2aicc[:,:,0,1]": [i, 4, self.gd2AICC[:, :, 0, 1]],
            # "dAICC_r1_r1 vs ..": [i, 5, dAICC_r1_r1, self.gd2AICC[1:-1, :, 0, 0]],
            # "dAICC_r1_r2 vs ..": [i, 6, dAICC_r1_r2, self.gd2AICC[1:, 1:, 0, 1]],
            #
            # "dAICC_r2": [i+1, 0, dAICC_r2],
            # "dAICC_r2_r1": [i+1, 1, dAICC_r2_r1],
            # "dAICC_r2_r2": [i+1, 2, dAICC_r2_r2],
            # "output_d2aicc[:,:,1,0]": [i+1, 3, self.gd2AICC[:, :, 1, 0]],
            # "output_d2aicc[:,:,1,1]": [i+1, 4, self.gd2AICC[:, :, 1, 1]],
            # "dAICC_r2_r1 vs ..": [i + 1, 5, dAICC_r2_r1, self.gd2AICC[1:, 1:, 1, 0]],
            # "dAICC_r2_r2 vs ..": [i + 1, 6, dAICC_r2_r2, self.gd2AICC[:, 1:-1, 1, 1]],
            #
            # "dRSS_r1": [j, 0, dRSS_r1],
            # "dRSS_r1_r1": [j, 1, dRSS_r1_r1],
            # "dRSS_r1_r2": [j, 2, dRSS_r1_r2],
            # "output_d2RSS[:,:,0,0]": [j, 3, self.gd2RSS[:, :, 0, 0]],
            # "output_d2RSS[:,:,0,1]": [j, 4, self.gd2RSS[:, :, 0, 1]],
            # "dRSS_r1_r1 vs ..": [j, 5, dRSS_r1_r1, self.gd2RSS[1:-1, :, 0, 0]],
            # "dRSS_r1_r2 vs ..": [j, 6, dRSS_r1_r2, self.gd2RSS[1:, 1:, 0, 1]],
            #
            # "dRSS_r2": [j+1, 0, dRSS_r2],
            # "dRSS_r2_r1": [j+1, 1, dRSS_r2_r1],
            # "dRSS_r2_r2": [j+1, 2, dRSS_r2_r2],
            # "output_d2RSS[:,:,1,0]": [j+1, 3, self.gd2RSS[:, :, 1, 0]],
            # "output_d2RSS[:,:,1,1]": [j+1, 4, self.gd2RSS[:, :, 1, 1]],
            # "dRSS_r2_r1 vs ..": [j + 1, 5, dRSS_r2_r1, self.gd2RSS[1:, 1:, 1, 0]],
            # "dRSS_r2_r2 vs ..": [j + 1, 6, dRSS_r2_r2, self.gd2RSS[:, 1:-1, 1, 1]],
            #
            # "dtrS_r1": [z, 0, dtrS_r1],
            # "dtrS_r1_r1": [z, 1, dtrS_r1_r1],
            # "dtrS_r1_r2": [z, 2, dtrS_r1_r2],
            # "output_d2trS[:,:,0,0]": [z, 3, self.gd2trS[:, :, 0, 0]],
            # "output_d2trS[:,:,0,1]": [z, 4, self.gd2trS[:, :, 0, 1]],
            # "dtrS_r1_r1 vs ..": [z, 5, dtrS_r1_r1, self.gd2trS[1:-1, :, 0, 0]],
            # "dtrS_r1_r2 vs ..": [z, 6, dtrS_r1_r2, self.gd2trS[1:, 1:, 0, 1]],
            #
            # "dtrS_r2": [z + 1, 0, dtrS_r2],
            # "dtrS_r2_r1": [z + 1, 1, dtrS_r2_r1],
            # "dtrS_r2_r2": [z + 1, 2, dtrS_r2_r2],
            # "output_d2trS[:,:,1,0]": [z + 1, 3, self.gd2trS[:, :, 1, 0]],
            # "output_d2trS[:,:,1,1]": [z + 1, 4, self.gd2trS[:, :, 1, 1]],
            # "dtrS_r2_r1 vs ..": [z + 1, 5, dtrS_r2_r1, self.gd2trS[1:, 1:, 1, 0]],
            # "dtrS_r2_r2 vs ..": [z + 1, 6, dtrS_r2_r2, self.gd2trS[:, 1:-1, 1, 1]]
        }

        #### cor_r1 = NUM.corrcoef(dAICC_r1.flatten(), -test1.dAICCgrid[:-1, :, 0].flatten())
        #### cor_r2 = NUM.corrcoef(dAICC_r2.flatten(), -test1.dAICCgrid[:, :-1, 1].flatten())

        fig, axs = plt.subplots(3, 7, figsize=(16, 7), sharey=False)
        fig.tight_layout(pad=3.0)
        origin = "lower"

        for k, v in vis_dict.items():
            if len(v) == 3:
                cmap, vmin, vmax = self.getVV(v[2])
                im00 = axs[v[0], v[1]].imshow(v[2], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)

                divider = make_axes_locatable(axs[v[0], v[1]])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im00, cax=cax)

                ### or by contour
                # axs[v[0], v[1]].contour(v[2], levels=10)

                if self.MGWR.neighbor_type == "distance":
                    axs[v[0], v[1]].set_xticklabels(NUM.round(self.raw_bandwidths, 2))
                    axs[v[0], v[1]].set_yticklabels(NUM.round(self.raw_bandwidths, 2))
                else:
                    axs[v[0], v[1]].set_xticklabels(self.raw_bandwidths)
                    axs[v[0], v[1]].set_yticklabels(self.raw_bandwidths)
                axs[v[0], v[1]].title.set_text(k)
                axs[v[0], v[1]].set_xlabel("r1")
                axs[v[0], v[1]].set_ylabel("r2")
            else:
                axs[v[0], v[1]].scatter(v[2], v[3], s=5)
                axs[v[0], v[1]].set_xlabel(v[4])
                axs[v[0], v[1]].set_ylabel(v[5])
            axs[v[0], v[1]].set_aspect("auto")

        p0 = pdf.PdfPages(self.file_name)
        p0.savefig()
        p0.close()

if __name__ == '__main__':
    test_output_dir = os.path.join(os.path.dirname(__file__), r"test\test_output")

    x, y, coords, data_path = get_georgia_data(nx=2)
    nedge = 30

    ### distance - bisquare
    m1 = MGWR(x, y, coords, data_path=data_path, neighbor_type="distance", kernel="bisquare", disp=False, poly=5)
    mm1 = gridMGWR(m1, nedge=nedge,
                              txt_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_distance_bisquare.txt"),
                              file_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_distance_bisquare.pdf"))
    mm1.run()
    print(mm1.message)
    mm1.plot()
    mm1.storeResults()

    ### distance - gaussian
    m2 = MGWR(x, y, coords, data_path=data_path, neighbor_type="distance", kernel="gaussian", disp=False, poly=5)
    mm2 = gridMGWR(m2, nedge=nedge,
                              txt_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_distance_gaussian.txt"),
                              file_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_distance_gaussian.pdf"))
    mm2.run()
    print(mm2.message)
    mm2.plot()
    mm2.storeResults()

    ### neighbor - bisquare
    m3 = MGWR(x, y, coords, data_path=data_path, neighbor_type="neighbor", kernel="bisquare", disp=False, poly=5)
    mm3 = gridMGWR(m3, nedge=nedge,
                     txt_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_neighbor_bisquare.txt"),
                     file_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_neighbor_bisquare.pdf"))
    mm3.run()
    print(mm3.message)
    mm3.plot()
    mm3.storeResults()

    ### neighbor - gaussian
    m4 = MGWR(x, y, coords, data_path=data_path, neighbor_type="neighbor", kernel="gaussian", disp=False, poly=5)
    mm4 = gridMGWR(m4, nedge=nedge,
                     txt_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_neighbor_gaussian.txt"),
                     file_name=os.path.join(test_output_dir, "test_mgwr_Jacobian_neighbor_gaussian.pdf"))

    mm4.run()
    print(mm4.message)
    mm4.plot()
    mm4.storeResults()
