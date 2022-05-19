"""create scenarios for multicode example

note that although there is only one scenario, we could easily add
more by extending the lists below. In general the RCE could change as well
and thus for each scenario we store the filepath and the unit length `a` which
is needed for some operations (i.e. to define the coarse grid for the
3x3 block used in the offline phase).
"""
import yaml
from pathlib import Path


example = Path(__file__).parent
scenarios = example / "scenarios.yml"

disc = [5]
degree = [2]
basis_type = ["empirical", "hierarchical"]

s = {}
sid = 0

for d in disc:
    for deg in degree:
        for basis in basis_type:
            rce_grid = example / "data" / f"rce_{sid}.xdmf"
            s[sid] = {
                "disc": d,
                "degree": deg,
                "basis_type": basis,
                "rce": {"xdmf": rce_grid.as_posix(), "a": 1.0},
            }
            sid += 1

with open(scenarios, "w") as out:
    yaml.safe_dump(s, out)
