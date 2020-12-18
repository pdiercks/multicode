"""task definition for test example"""

#  from doit import create_after
#  from doit.action import CmdAction
from doit.tools import run_once
from pathlib import Path
import yaml

example = Path(__file__).parent
multicode = example.absolute().parent
preprocessing = multicode / "preprocessing"
scenarios = example / "scenarios.yml"


def task_create_scenarios():
    """create all scenarios for this example"""

    def create(targets):
        """create the scenario for this example

        note that although there is only one scenario, we could easily add
        more by extending the lists below. In general the RVE could change as well
        and thus for each scenario we store the filepath and the unit length `a` which
        is needed for some operations (i.e. to define the coarse grid for the
        3x3 block used in the offline phase).
        """
        disc = [5]
        degree = [2]
        basis_type = ["empirical"]
        s = {}
        sid = 0
        for d in disc:
            for deg in degree:
                for basis in basis_type:
                    rve_grid = example / f"rve_{sid}.xdmf"
                    s[sid] = {
                        "disc": d,
                        "degree": deg,
                        "basis_type": basis,
                        "rve": {"xdmf": rve_grid.as_posix(), "a": 1.0},
                    }
                    sid += 1
        with open(targets[0], "w") as out:
            yaml.safe_dump(s, out)

    return {
        "actions": [create],
        "targets": [scenarios],
        "uptodate": [run_once],
        "clean": True,
    }
