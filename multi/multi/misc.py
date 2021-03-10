"""miscellaneous helpers"""

from pathlib import Path
import dolfin as df
import numpy as np
import yaml
from multi import Domain, ResultFile, LinearElasticityProblem
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace


def get_solver(yml_file):
    default_solver = {
        "krylov_solver": {
            "relative_tolerance": 1.0e-9,
            "absolute_tolerance": 1.0e-12,
            "maximum_iterations": 1000,
        },
        "solver_parameters": {"linear_solver": "default", "preconditioner": "default"},
    }

    if yml_file is not None:
        assert Path(yml_file).suffix in (".yml", ".yaml")
        try:
            with open(yml_file, "r") as f:
                solver = yaml.safe_load(f)
            return solver

        except FileNotFoundError:
            print(
                f"File {yml_file} could not be found. Using default solver settings ..."
            )
            return default_solver
    else:
        return default_solver


def make_mapping(sub_space, super_space):
    """get map from sub to super space

    Parameters
    ----------
    sub_space
        A dolfin.FunctionSpace
    super_space
        A dolfin.FunctionSpace

    Returns
    -------
    The dofs of super_space located at entities in sub_space.

    Note: This only works for conforming meshes.
    """
    f = df.Function(super_space)
    f.vector().set_local(super_space.dofmap().dofs())
    If = df.interpolate(f, sub_space)
    return (If.vector().get_local() + 0.5).astype(int)


def compute_norms(
    rvemeshfile,
    degree,
    material,
    dofmap,
    udns,
    urb,
    basis,
    product=None,
    output=None,
    output_strain=False,
):
    """compute absolute error (ROM wrt DNS) and dns norm wrt given inner product

    Parameters
    ----------

    rvemeshfile : Path, str
        Filepath to rve mesh.
    degree : int
        Degree of FE space.
    material : dict
        Material metadata.
    dofmap : multi.DofMap
        Multiscale dofmap.
    udns : df.Function
        The direct numerical simulation.
    urb : np.ndarray
        The ROM solution.
    basis : np.ndarray
        The reduced basis.
    product : optional, str
        The inner product to use.
    output : optional, str
        FilePath to write functions to.
    output_strain : optional, bool
        If True, write absolute strain error
        (If output is not None).

    Returns
    -------
    norms : dict
        absolute error and dns norms

    """

    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]
    product_to_use = product

    norms = {"abs_err": [], "dns": []}
    for cell_index, cell in enumerate(dofmap.cells):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        omega = Domain(
            rvemeshfile, cell_index, subdomains=True, translate=df.Point(offset)
        )
        V = df.VectorFunctionSpace(omega.mesh, "CG", degree)
        problem = LinearElasticityProblem(
            omega, V, E=E, NU=NU, plane_stress=plane_stress
        )
        dns = df.interpolate(udns, V)
        if product_to_use:
            p_mat = problem.get_product(name=product_to_use, bcs=False)
            product = FenicsMatrixOperator(p_mat, V, V)
        else:
            product = None
        dofs = dofmap.cell_dofs(cell_index)
        u_rb = df.Function(V)
        u_rb.vector().set_local(basis.T @ urb[dofs])

        S = FenicsVectorSpace(V)
        DNS = S.make_array([dns.vector()])
        URB = S.make_array([u_rb.vector()])
        ERR = DNS - URB
        dns_norm = DNS.norm(product)
        err_norm = ERR.norm(product)
        norms["dns"].append(dns_norm)
        norms["abs_err"].append(err_norm)

        if output is not None and output.suffix == ".xdmf":
            aerr = df.Function(V)
            rerr = df.Function(V)
            abs_values = dns.vector().get_local() - u_rb.vector().get_local()
            aerr.vector().set_local(abs_values)
            rerr.vector().set_local(abs_values / dns.vector().get_local())

            output = Path(output)
            name = output.stem
            rve_results = output.parent / (name + f"_{cell_index}.xdmf")

            xdmf = ResultFile(rve_results)
            xdmf.add_function(u_rb, name="u_rb")
            xdmf.add_function(dns, name="dns")
            xdmf.add_function(aerr, name="aerr")
            xdmf.add_function(rerr, name="rerr")

            if output_strain:
                VDG = df.VectorFunctionSpace(omega.mesh, "DG", degree - 1, dim=3)
                e = df.sym(df.grad(aerr))
                evoigt = df.as_vector([e[0, 0], e[1, 1], e[0, 1]])
                strain = df.Function(VDG)
                lp = LocalProjector(evoigt, VDG, df.dx)
                lp(strain)
                xdmf.add_function(strain, name="strain abs err")

            xdmf.write(0)
            xdmf.close()

    return norms


class LocalProjector:
    def __init__(self, expr, V, dxm):
        """
        expr:
            expression to project
        V:
            (quadrature) function space
        dxm:
            dolfin.Measure("dx") that matches V
        """
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        a_proj = df.inner(dv, v_) * dxm
        b_proj = df.inner(expr, v_) * dxm
        self.solver = df.LocalSolver(a_proj, b_proj)
        self.solver.factorize()

    def __call__(self, u):
        """
        u:
            function that is filled with the solution of the projection
        """
        self.solver.solve_local_rhs(u)


def read_basis(npz_filename, modes_per_edge=None):
    """read multiscale basis

    Parameters
    ----------
    npz_filename : str, Path
        FilePath to multiscale basis.
    modes_per_edge : int, optional
        Number of basis functions per edge.

    Returns
    -------
    np.ndarray

    """
    r = []
    edges = ["b", "r", "t", "l"]

    data = np.load(npz_filename)
    max_modes = min([len(data[k]) for k in edges])
    if modes_per_edge is not None and modes_per_edge < max_modes:
        m = modes_per_edge
    else:
        m = max_modes

    r.append(data["phi"])
    for key in edges:
        r.append(data[key][:m])

    return np.vstack(r)


def select_modes(basis, modes_per_edge, max_modes):
    """select modes according to multi.DofMap

    Parameters
    ----------
    basis : np.ndarray
        The multiscale basis used.
    modes_per_edge : int
        Number of modes per edge.
    max_modes : int
        Maximum number of modes per edge.

    Returns
    -------
    basis : np.ndarray
        Subset of the full basis.

    """

    offset = 0
    coarse = [i for i in range(8)]
    offset += len(coarse)
    bottom = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    right = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    top = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    left = [offset + i for i in range(modes_per_edge)]
    ind = coarse + bottom + right + top + left
    return basis[ind]
