"""task definition for test example"""

from doit import create_after
from doit.action import CmdAction
from doit.tools import run_once
from pathlib import Path
import yaml

example = Path(__file__).parent
multicode = example.absolute().parent
preprocessing = multicode / "preprocessing"
scenarios = example / "scenarios.yml"


def task_mkdirs():
    """create directories"""

    def mkdir(targets):
        for t in targets:
            Path(t).mkdir(parents=True, exist_ok=False)

    return {
        "actions": [mkdir],
        "targets": [(example / p) for p in ["data", "results"]],
        "uptodate": [run_once],
        "clean": True,
    }


@create_after(executed="mkdirs")
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
                    rve_grid = example / "data" / f"rve_{sid}.xdmf"
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


@create_after(executed="create_scenarios")
def task_rve_grid():
    """create RVE fine grid"""
    script = preprocessing / "rve_type_01.py"
    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    for i, scenario in s.items():
        target = Path(scenario["rve"]["xdmf"])
        N = int((scenario["disc"] + 1) / 2)
        cmd = f"python {script} 0 0 1 1 0.2 {N} --output={target}"
        yield {
            "name": f"{i}",
            "file_dep": [scenarios, script],
            "actions": [CmdAction(cmd)],
            "targets": [target, target.with_suffix(".h5")],
            "clean": True,
            "task_dep": ["create_scenarios"],
        }


@create_after(executed="create_scenarios")
def task_rve_edges():
    """create meshes for the boundary of the RVE"""
    script = preprocessing / "rve_edges.py"
    edges = ["bottom", "right", "top", "left"]

    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    for i, scenario in s.items():
        N = scenario["disc"]
        rve = Path(scenario["rve"]["xdmf"])
        # note target definition according to rve_edges.py
        targets = [rve.parent / (rve.stem + f"_{e}.xdmf") for e in edges]
        cmd = f"python {script} {rve} --transfinite={N}"
        yield {
            "name": f"{i}",
            "file_dep": [script, scenarios, rve],
            "actions": [CmdAction(cmd)],
            "targets": targets + [t.with_suffix(".h5") for t in targets],
            "clean": True,
        }


@create_after(executed="create_scenarios")
def task_coarse_grid():
    """create coarse grid for this example which is a 3x3 block of RVEs"""
    script = preprocessing / "rectangle.py"

    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    N = 3
    scenario = s[0]
    # RVE with varying unit length is out of scope
    x = scenario["rve"]["a"] * N
    target = example / "data" / "coarse_block.msh"
    cmd = f"python {script} {x} {x} {N} {N} 1 --transfinite --quads"
    cmd += f" --output={target}"
    return {
        "file_dep": [script, scenarios],
        "actions": [CmdAction(cmd)],
        "targets": [target],
        "clean": True,
    }


@create_after(executed="create_scenarios")
def task_fine_grid():
    """create fine grid for global structure"""
    script = preprocessing / "soi.py"

    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    coarse_grid = example / "data" / "coarse_block.msh"
    for i, scenario in s.items():
        rve = Path(scenario["rve"]["xdmf"])
        fine_grid = example / "data" / f"block_{i}.xdmf"
        cmd = f"python {script} {coarse_grid} {rve} --tdim=2 --gdim=2 --prune_z_0"
        cmd += f" --output={fine_grid}"
        yield {
            "name": f"{i}",
            "file_dep": [script, coarse_grid, rve, scenarios],
            "actions": [CmdAction(cmd)],
            "targets": [fine_grid, fine_grid.with_suffix(".h5")],
            "clean": True,
        }


@create_after(executed="create_scenarios")
def task_empirical_basis():
    """construct empirical basis for the given RVE"""
    mat = example / "material.yml"
    solver = example / "solver.yml"

    with open(scenarios, "r") as ins:
        s = yaml.safe_load(ins)

    edges = ["bottom", "right", "top", "left"]
    for i, scenario in s.items():
        block = example / "data" / f"block_{i}.xdmf"
        rve = Path(scenario["rve"]["xdmf"])
        a = scenario["rve"]["a"]
        rve_edges = [rve.parent / (rve.stem + f"_{e}.xdmf") for e in edges]
        degree = scenario["degree"]
        basis = scenario["basis_type"]
        script = multicode / f"{basis}_basis.py"
        targets = [
            example / "results" / f"basis_{i}.npy",
            example / "results" / f"edge_basis_{i}.npy",
            example / "results" / f"testing_set_proj_err_{i}.txt",
        ]
        cmd = f"python {script} {block} {rve} {a} {degree} {mat} --training-set=delta"
        cmd += f" --output={targets[0]} --chi={targets[1]} --projerr={targets[2]}"
        cmd += f" --solver={solver} --test --check-interface=1e-8"
        yield {
            "name": f"{i}",
            "actions": [CmdAction(cmd)],
            "targets": targets,
            "file_dep": [scenarios, block, rve, *rve_edges, script, mat, solver],
            "clean": True,
        }
