### How to start 
- You should be able to run the code with default conda environment. If not, try to create Python environment using [gmgwr_env.yml](gmgwr_env.yml) following [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) guide.
- As the first trial, run [mgwr/tests/main.py](mgwr/tests/main.py), see more at [mgwr/tests/README.md](mgwr/tests/README.md).  
  
### Structure 
- The core functions locate in [mgwr](mgwr)
- Some testing scripts and sample results locates [mgwr/tests](mgwr/tests)

### Benchmark 
https://github.com/pysal/mgwr/
- equal_interval() in search.py has bug. when running MGWR with .search(search_method="interval"), l_bound and u_bound is not correctly passed and get error TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
- In this package, bw = 100 means a local regression with 100 neighboring points including itself. In Pro GWR, bw = 101 means 100 neighboring points + 1. 

### Reference for Trust Newton Algorithm
Julia 
- http://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/newton_trust_region/
- https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

Trust Region Newton Method for Large-Scale Logistic Regression 
- https://jmlr.csail.mit.edu/papers/volume9/lin08b/lin08b.pdf

NEWTON’S METHOD FOR LARGE BOUND-CONSTRAINED OPTIMIZATION PROBLEMS 
- https://www.csie.ntu.edu.tw/~cjlin/papers/tron.pdf
- https://www.mcs.anl.gov/~anitescu/CLASSES/2012/LECTURES/S310-2012-lect5.pdf

**saddle point**

- https://people.orie.cornell.edu/dsd95/SIOPTsaddles.pdf
- https://arxiv.org/pdf/2006.01512v3.pdf
- https://github.com/hphuongdhsp/Q-Newton-method/blob/master/src/functionsDirectRun.py
