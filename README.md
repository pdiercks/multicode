# multicode

A small library to construct (localized) basis functions in the context of a variational multiscale method to model heterogeneous structures.
The library contains 
* the python package `multi` which handles repeatedly upcoming operations,
* scripts / programs which can be used to generate meshes and construct a reduced basis.

## Installation
It is recommended to install required dependencies via conda.
* docopt
* fenicsx (dolfinx, basix, ufl, ffcx)
* gmsh
* h5py
* meshio
* matplotlib (note that the `pgf` backend is used and requires a texlive installation)
* pyaml
* pymor
* pyvista
* pytest
* sympy

The source code can be installed via `pip`.
```
git clone https://git.bam.de/mechanics/pdiercks/multicode.git && cd multi
$PYTHON -m pip install <path/to/package/directory>
```

## Usage

The following tasks can be completed with this library:
* basis construction (`generate_basis.py`),
* computation of homogenized material parameters (`homogenization.py`),
* common preprocessing tasks related to the above (see `preprocessing`), 
* common postprocessing tasks (for specific file types and data structures, see `postprocessing`).
