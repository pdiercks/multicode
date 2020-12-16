"""task definition for test example"""

from doit.tools import run_once
from pathlib import Path
import yaml

example = Path(__file__).parent
preprocessing = example.absolute().parent / "preprocessing"


def task_create_scenarios():
    """create all scenarios for this example, where each
    scenario consists of discretization, degree and basis type"""
    targets = [example / "scenarios.yml"]

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
                    s[sid] = {
                            "disc": d, "degree": deg, "basis_type": basis
                            }
                    sid += 1
        with open(targets[0], "w") as out:
            yaml.safe_dump(s, out)

    return {
        "actions": [create_scenarios],
        "targets": targets,
        "uptodate": [run_once],
        "verbosity": 2,
        }



def task_rve_grid():
    """create RVE fine grid"""
    pass
    #  script = preprocessing / "rve_type_01.py"
