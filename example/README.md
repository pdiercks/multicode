# Example

This example is used to demonstrate the use of the `multicode` library as well as to test it.
All tasks for this example are defined in `dodo.py` and can be run by running `doit` on the command line. 

## Workflow

To use this library for multiscale simulations proceed as follows:
1. definition of the RVE to use (metadata `rve.yml` and the fine grid `rve.xdmf`),
2. definiton of scenario(s) for offline and online phase,
3. creation of all grids:
    * offline: coarse and fine grid for 3x3 patch made of RVE defined in step 1.,
    * online: coarse grid of the structure of interest and optionally fine grid of the structure of interest for error computation,
4. construction of a basis functions according to the scenario(s).
