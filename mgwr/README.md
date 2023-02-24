### Note
- refer to [mgwr_esri/README.md](mgwr_esri/README.md) for setup. 

### Structure 
- ``mgwr.py`` includes the MGWR class. 
  - ``scipy_optimize()`` is the leading function. 
  - A few functions are unique for gradient-MGWR: 
    - ``scale_coord(), scale_distance()`` scales the coordinates/distance (to unitless) before optimization. the scale is controlled by `coords_scale_factor`.
    - ``reverse_scaling_distance()`` is the opposite process, to provide meaning distance (with unit) after optimization. 
    - ``dist_nb_smoothing()`` builds projection between number of neighbors and distance. the key idea is introduced in the paper. some of the relevant codes are in ``kernels.py``. 
  - Some functions do not have to be strictly followed:
    - this includes ``set_up(), call_back(), set_bound(), set_start(), careate_output_table(), summary_excel(), result_store()``. we can use existing code in MGWR class in ArcGIS Pro. 
    - e.g. ``set_start()`` obtains the starting bandwidth using gradient-GWR, but we can use our existing GWR model in ArcGIS Pro. 
  - Other functions are no longer used. 
    - ``newton.py()`` does not use scipy module but the optimization performance is bad. 
    - ``pysal_exec()`` calls pysal library. but MGWR in pysal is not parallel so we don't use it anymore. 
  - This module can execute GWR, but scripts relating to GWR are not well organized and it is slow. we don't have to use those. 
- ``gradients.py`` holds the core algorithm of gradient derivation. 
  - ``gradient_mgwr(), loop_nk(), matrix_calc()`` should be developed. 
  - ignore all other functions. ``gradient_gwr()`` is functioning, but it's slow.
- ``kernels.py`` holds the algorithm to compute kernels. 
  - we need to develop this because they includes some gradient derivation. 
  - we have four kernel types here (distance/neighbor + guassian/bisquare). part of gradient are derived here, based on the chain rule. 
- ignore other scripts ``mgwr_esri/gradients_func_ij.py``, ``mgwr_esri/gradients_multi.py``, ``optimize.py``.