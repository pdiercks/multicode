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
    """create all scenarios for this example,
    where each scenario consists of discretization, degree and basis type"""

    def create_scenarios(targets):
        # discretization with 5 points per edge
        disc = [5]
        degree = [2]
        basis_type = ["empirical"]
        s = {}
        sid = 0
        for d in disc:
            for deg in degree:
                for basis in basis_type:
                    rve = example / f"rve_{sid}.xdmf"
                    block = example / f"block_{sid}.xdmf"
                    s[sid] = {
                        # FIXME figure out what should be part of scenarios and what not ...
                        "disc": d,
                        "degree": deg,
                        "basis_type": basis,
                        "rve": rve.as_posix(),
                        "block": block.as_posix(),  # block mesh for offline phase
                        "a": 1.0,  # unit length of the RVE
                    }
                    sid += 1
        with open(targets[0], "w") as out:
            yaml.safe_dump(s, out)

    return {
        "actions": [create_scenarios],
        "targets": [scenarios],
        "uptodate": [run_once],
        "clean": True,
    }
