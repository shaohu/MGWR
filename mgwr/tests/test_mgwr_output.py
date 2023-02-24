import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mgwr import *
from mgwr.tests.data import *

test_output_dir = os.path.join(os.path.dirname(__file__), r"test\test_output")

x, y, coords, data_path = get_georgia_data(nx=6)
neighbor_type = "distance"
kernel_type = "bisquare"

m1 = MGWR(x, y, coords, data_path=data_path, neighbor_type=neighbor_type, kernel=kernel_type)
m1.scipy_optimize(method="TNC")
m1.result_store(print_output = True, result_file = os.path.join(test_output_dir, "test_mgwr_output.csv"))

# MGWR bandwidths
# --------------------------------------------------------------------------------
# Variable                  Bandwidth      ENP_j   Adj t-val(95%)            DoD_j
# X0                       342970.342      2.038            0.025            2.270
# X1                       240222.090      3.553            0.014            2.483
# X2                       338741.323      2.276            0.022            2.314
# X3                       615942.647      1.334            0.037            2.098
# X4                      1229586.808      1.096            0.046            2.015
# X5                       424954.986      1.263            0.040            2.075
#
# Diagnostic Information
# --------------------------------------------------------------------------------
# Residual sum of squares:                                                  48.541
# Effective number of parameters (trace(S)):                                11.560
# Degree of freedom (n - trace(S)):                                        147.440
# Sigma estimate:                                                            0.574
# Log-likelihood:                                                         -131.285
# AIC:                                                                     287.691
# AICc:                                                                    290.033
# R2:                                                                        0.695
# Adj. R2:                                                                   0.671
#
# Summary Statistics For MGWR Parameter Estimates
# --------------------------------------------------------------------------------
# Variable                        Mean        STD        Min     Median        Max
# --------------------      ---------- ---------- ---------- ---------- ----------
# X0                             0.002      0.101     -0.187      0.025      0.155
# X1                             0.417      0.262      0.037      0.472      0.754
# X2                             0.112      0.118     -0.075      0.084      0.364
# X3                            -0.326      0.026     -0.376     -0.328     -0.267
# X4                             0.003      0.001      0.000      0.003      0.005
# X5                            -0.252      0.085     -0.399     -0.267     -0.060
# ================================================================================

m1.pysal_exec()

# Multi-Scale Geographically Weighted Regression (MGWR) Results
# ---------------------------------------------------------------------------
# Spatial kernel:                                          Fixed bisquare
# Criterion for optimal bandwidth:                                       AICc
# Score of Change (SOC) type:                                     Smoothing f
# Termination criterion for MGWR:                                       1e-05
#
# MGWR bandwidths
# ---------------------------------------------------------------------------
# Variable             Bandwidth      ENP_j   Adj t-val(95%)   Adj alpha(95%)
# X0                       1.960      3.162            2.440            0.016
# X1                       1.740      3.957            2.523            0.013
# X2                       9.460      1.040            1.992            0.048
# X3                       2.840      1.961            2.255            0.025
# X4                       9.460      1.060            2.000            0.047
# X5                       9.460      1.034            1.990            0.048
#
# Diagnostic information
# ---------------------------------------------------------------------------
# Residual sum of squares:                                             48.720
# Effective number of parameters (trace(S)):                           12.214
# Degree of freedom (n - trace(S)):                                   146.786
# Sigma estimate:                                                       0.576
# Log-likelihood:                                                    -131.577
# AIC:                                                                289.583
# AICc:                                                               292.177
# BIC:                                                                330.135
# R2                                                                    0.694
# Adjusted R2                                                           0.668
#
# Summary Statistics For MGWR Parameter Estimates
# ---------------------------------------------------------------------------
# Variable                   Mean        STD        Min     Median        Max
# -------------------- ---------- ---------- ---------- ---------- ----------
# X0                       -0.010      0.113     -0.229      0.000      0.197
# X1                        0.428      0.238      0.026      0.481      0.715
# X2                        0.096      0.002      0.093      0.096      0.100
# X3                       -0.284      0.033     -0.401     -0.269     -0.244
# X4                       -0.052      0.001     -0.053     -0.052     -0.051
# X5                       -0.181      0.001     -0.183     -0.181     -0.179
# ===========================================================================
