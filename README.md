# multicode

A small library to construct (localized) basis functions in the context of a variational multiscale method to model heterogeneous structures.

## Installation

It is recommended to install required dependencies via conda.
* fenicsx (dolfinx, basix, ufl, ffcx)
* gmsh
* meshio
* matplotlib (note that the `pgf` backend is used and requires a texlive installation)
* pyaml
* pyvista
* pytest
* sympy
* pymor

The source code can be installed via `pip`.
```
git clone https://git.bam.de/mechanics/pdiercks/multicode.git multi && cd multi
$PYTHON -m pip install .
```
