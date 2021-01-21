# multicode

A small library to construct (localized) basis functions in the context of a variational multiscale method to model heterogeneous structures.
The library contains 
* the python package `multi` which handles repeatedly upcoming operations,
* scripts / programs which can be used to generate meshes and construct a reduced basis.

## Installation

To use the library you need to install all software dependencies which are listed in the below command(s) (using conda and pip).
```
conda create -n <env-name> fenics=2019.1.0 gmsh=4.6.0 doit meshio=4.3.1 matplotlib pyaml line_profiler pymor=2020.1.2 pytest
conda activate <env-name> && conda install -c pdiercks pygmsh
```
Or:
```
conda env create -f env.yml
```
Note that you need to change the `prefix` in `env.yml`
(`env.yml` was created by `conda env export --from-history > env.yml`).

### Additional packages

Unfortunately not all required packages are available from `anaconda.org`.
These are installed via pip *after* everything else was installed successfully.

#### fenics_helpers
```
pip3 install --user git+https://github.com/BAMresearch/fenics_helpers.git
```

#### multi
```
git clone https://git.bam.de/mechanics/pdiercks/multicode.git && cd multi
pip install .
```

#### plotstuff
```
git clone https://git.bam.de/mechanics/cpohl/plotstuff.git && cd plotstuff
pip install .
```

## Usage

The following tasks can be completed with this library:
* basis construction (`generate_basis.py`),
* computation of homogenized material parameters (`homogenization.py`),
* common preprocessing tasks related to the above (see `preprocessing`), 
* common postprocessing tasks (for specific file types and data structures, see `postprocessing`).
