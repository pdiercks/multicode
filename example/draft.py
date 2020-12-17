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
metadata = example / "rve.yml"


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
                    s[sid] = {"disc": d, "degree": deg, "basis_type": basis}
                    sid += 1
        with open(targets[0], "w") as out:
            yaml.safe_dump(s, out)

    return {
        "actions": [create_scenarios],
        "targets": [scenarios],
        "uptodate": [run_once],
        "clean": True,
    }


def task_rve_metadata():
    """add rve metadata for each scenario"""

    def create_metadata(dependencies):
        with open(dependencies[0], "r") as instream:
            s = yaml.safe_load(instream)

        r = {}
        for i in range(len(s)):
            xdmf = example / f"rve_{i}.xdmf"
            s[i].update({"a": 1.0, "xdmf": xdmf.as_posix()})

        with open(dependencies[0], "w") as out:
            yaml.safe_dump(r, out)

    return {
        "file_dep": [scenarios],
        "actions": [create_metadata],
    }


@create_after(executed="create_scenarios")
def task_rve_grid():
    """create RVE fine grid"""
    script = preprocessing / "rve_type_01.py"
    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    for i, scenario in s.items():
        target = example / f"rve_{i}.xdmf"
        N = int((scenario["disc"] + 1) / 2)
        cmd = f"python {script} 0 0 1 1 0.2 {N} --output={target}"
        yield {
            "name": f"{i}",
            "file_dep": [scenarios, script],
            "actions": [CmdAction(cmd)],
            "targets": [target, target.with_suffix(".h5")],
            "clean": True,
            "task_dep": ["create_scenarios"],
            "verbosity": 2,
        }


@create_after(executed="rve_metadata")
def task_rve_edges():
    """create meshes for the boundary of the RVE"""
    script = preprocessing / "rve_edges.py"
    edges = ["bottom", "right", "top", "left"]

    with open(metadata, "r") as md:
        rvemd = yaml.safe_load(md)
    with open(scenarios, "r") as instream:
        s = yaml.safe_load(instream)

    for i, scenario in s.items():
        N = rvemd[i]["disc"]  # unit length possibly depending on scenario
        rve = Path(rvemd[i]["xdmf"])
        # note target definition according to rve_edges.py
        targets = [rve.parent / (rve.stem + f"_{e}.xdmf") for e in edges]
        cmd = f"python {script} {rve} --transfinite={N}"
        yield {
            "name": f"{i}",
            "file_dep": [script, metadata, scenarios, rve],
            "actions": [CmdAction(cmd)],
            "targets": targets + [t.with_suffix(".h5") for t in targets],
            "clean": True,
            "verbosity": 2,
        }




@create_after(executed="rve_metadata")
def task_coarse_grid():
    """create coarse grid for this example which is a 3x3 block of RVEs"""
    script = preprocessing / "rectangle.py"

    with open(metadata, "r") as md:
        rvemd = yaml.safe_load(md)

    N = 3
    for i, rve in rvemd.items():
        target = example / f"coarse_block_{i}.msh"
        x = rve["a"] * N
        cmd = f"python {script} {x} {x} {N} {N} 1 --transfinite --quads"
        cmd += f" --output={target}"
        return {
            "file_dep": [script, metadata],
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

    for i, scenario in s.items():
        coarse_grid = example / f"coarse_block_{i}.msh"
        rve = example / f"rve_{i}.xdmf"
        fine_grid = example / f"block_{i}.xdmf"
        cmd = f"python {script} {coarse_grid} {rve} --tdim=2 --gdim=2 --prune_z_0"
        cmd += f" --output={fine_grid}"
        yield {
            "name": f"{i}",
            "file_dep": [script, coarse_grid, rve, scenarios],
            "actions": [CmdAction(cmd)],
            "targets": [fine_grid, fine_grid.with_suffix(".h5")],
            "clean": True,
        }


#  @create_after(executed="rve_metadata")
#  def task_empirical_basis():
#      """construct empirical basis for the given RVE"""
#      mat = example / "material.yml"
#      solver = example / "solver.yml"
#
#      with open(scenarios, "r") as ins:
#          s = yaml.safe_load(ins)
#      with open(metadata, "r") as md:
#          r = yaml.safe_load(md)
#
#      edges = ["bottom", "right", "top", "left"]
#      for i, scenario in s.items():
#          block = example / f"block_{i}.xdmf"
#          rve = example / f"rve_{i}.xdmf"
#          rve_edges = [rve.parent / (rve.stem + f"_{e}.xdmf") for e in edges]
#          degree = scenario["degree"]
#          basis = scenario["basis_type"]
#          script = multicode / f"{basis}_basis.py"
#          targets = [
#              example / f"basis_{i}.npy",
#              example / f"edge_basis_{i}.npy",
#              example / f"testing_set_proj_err_{i}.txt",
#          ]
#          cmd = f"python {script} {block} {rve} {degree} --training-set=delta"
#          cmd += f" --output={targets[0]} --chi={targets[1]} --projerr={targets[2]}"
#          yield {
#              "name": f"{i}",
#              "actions": [CmdAction(cmd)],
#              "targets": targets,
#              "file_dep": [scenarios, metadata, block, rve, *rve_edges, script],
#              "clean": True,
#          }
