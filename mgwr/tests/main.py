import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

if __name__ == '__main__':
    results_table = pd.DataFrame(columns=['data', 'nSample', 'nVariate', 'mode', 'neighborType', 'kernel', 'method',
                                         'time', 'AICc', 'iteration', 'bandwidth', 'message'])

    test_output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    # test_output_dir = r"C:\temp"
    output_file = os.path.join(test_output_dir, r"main"+ str(np.random.random()) +".csv")

    x, y, coords, data_path = get_georgia_data(nx=2)
    for mode_ in ["mgwr"]:
        for neighbor_type_ in ["neighbor", "distance"]:
            for kernel_type_ in ["gaussian", "bisquare"]:
                for gwr_init_ in [False, True]:

                    if mode_ == "gwr" and gwr_init_ == True:
                        continue

                    md = MGWR(x, y, coords, neighbor_type=neighbor_type_, kernel=kernel_type_,
                              mode=mode_, disp=True, gwr_init=gwr_init_, data_path=data_path,
                              scale_coordinate=True)

                    md.newton(hess=False)
                    md.scipy_optimize(method="L-BFGS-B")
                    md.scipy_optimize(method="TNC")
                    md.scipy_optimize(method="trust-constr")

                    # these two function no longer work after using numba
                    # md.newton(hess=True)
                    # md.pysal_exec()

                    md.outputResults["gwr_init"] = gwr_init_
                    results_table = results_table.append(md.outputResults)
                    results_table.to_csv(output_file)

    print("output file is:  ", output_file)


