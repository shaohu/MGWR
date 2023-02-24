"""
outdated
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

test_output_dir = os.path.join(os.path.dirname(__file__), r"test\test_output")

x, y, coords, data_path = get_georgia_data(nx=2)
neighbor_type = "distance"
kernel_type = "bisquare"

### search
m1 = MGWR(x, y, coords, data_path=data_path, neighbor_type=neighbor_type, kernel=kernel_type, aicc_tol = 1e-8, disp=True)
if neighbor_type == "neighbor":
    m1.x0 = [40, 40]
    m1.newton(learning_rate=10)
    m1.newton(learning_rate=40)
else:
    m1.newton()
    m1.newton(learning_rate=0.5)

### background AICc contour
mm1 = gridMGWR(m1, nedge=30)
mm1.run()

### visualize
p0 = pdf.PdfPages(os.path.join(test_output_dir, "test_mgwr_search_"+neighbor_type+".pdf"))

Y, X = NUM.meshgrid(mm1.bandwidths, mm1.bandwidths)

nplot = m1.outputResults.shape[0]
fig, axs = plt.subplots(1, nplot, figsize=(10, 6))

for i in range(nplot):
    pcontour = axs[i].contour(X, Y, mm1.gAICC, levels=30)
    plt.clabel(pcontour, inline=1, fontsize=8)

    axs[i].set_xlabel("r1")
    axs[i].set_ylabel("r2")
    axs[i].set_title(m1.outputResults.method[i])

    try:
        bws = m1.outputResults.bandwidth_history[i]
        for j in range(1, len(bws)):
            axs[i].annotate('', xy=(bws[j]), xytext=(bws[j - 1]),
                            arrowprops={'arrowstyle': '->', 'color': 'red', 'lw': 1},
                            va='center', ha='center')
    except:
        if m1.neighbor_type == "distance":
            axs[i].scatter(x=m1.outputResults.bandwidth_scaled[i][0],
                           y=m1.outputResults.bandwidth_scaled[i][1], c="orange")
        else:
            axs[i].scatter(x=m1.outputResults.bandwidth[i][0],
                           y=m1.outputResults.bandwidth[i][1], c="orange")
p0.savefig()
p0.close()
