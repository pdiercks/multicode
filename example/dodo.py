"""task definition for test example"""

from doit.action import CmdAction
from doit.tools import run_once
from pathlib import Path
import yaml

example = Path(__file__).parent
preprocessing = example.absolute().parent / "preprocessing"
scenarios = [example / "scenarios.yml"]


def task_create_scenarios():
    """create all scenarios for this example, where each
    scenario consists of discretization, degree and basis type"""

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
                    s[sid] = {"disc": d, "degree": deg, "basis_type": basis}
                    sid += 1
        with open(targets[0], "w") as out:
            yaml.safe_dump(s, out)

    return {
        "actions": [create_scenarios],
        "targets": scenarios,
        "uptodate": [run_once],
        "verbosity": 2,
    }


def task_rve_grid():
    """create RVE fine grid"""
    script = preprocessing / "rve_type_01.py"
    with open(scenarios[0], "r") as instream:
        s = yaml.safe_load(instream)

    for i, scenario in s.items():
        targets = [example / f"rve_{i}.xdmf"]
        N = int((scenario["disc"] + 1) / 2)
        cmd = f"python {script} 0 0 1 1 0.2 {N} --output={targets[0]}"
        yield {
            "name": f"{i}",
            "file_dep": scenarios + [script],
            "actions": [CmdAction(cmd)],
            "targets": targets,
            "clean": True,
        }


def task_rve_metadata():
    """define rve metadata for each scenario"""
    targets = [example / "rve.yml"]

    def create_metadata(targets):
        with open(scenarios[0], "r") as instream:
            s = yaml.safe_load(instream)

        r = {}
        for i, scenario in s.items():
            xdmf = example / f"rve_{i}.xdmf"
            r[i] = {"disc": scenario["disc"], "a": 1.0, "xdmf": xdmf.as_posix()}
        with open(targets[0], "w") as out:
            yaml.safe_dump(r, out)

    return {
        "targets": targets,
        "actions": [create_metadata],
        "uptodate": [run_once],
        "verbosity": 2,
        }


