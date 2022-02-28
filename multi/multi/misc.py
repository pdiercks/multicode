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

    # instantiate mesh and function space in origin
    reference_domain = Domain(rvemeshfile, 999, subdomains=True)
    V = df.VectorFunctionSpace(reference_domain.mesh, "CG", degree)
    # functio space for stress/strain output
    VDG = df.VectorFunctionSpace(reference_domain.mesh, "DG", degree - 1, dim=3)
    problem = LinearElasticityProblem(
        reference_domain, V, E=E, NU=NU, plane_stress=plane_stress
    )

    if product_to_use:
        p_mat = problem.get_product(name=product_to_use, bcs=False)
        product = FenicsMatrixOperator(p_mat, V, V)
    else:
        product = None

    u_rb = df.Function(V)
    norms = {"abs_err": [], "dns": []}
    for cell_index, cell in enumerate(dofmap.cells):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        reference_domain.mesh.translate(df.Point(offset))

        dns = df.interpolate(udns, V)
        dofs = dofmap.cell_dofs(cell_index)
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
            rerr.vector().set_local(abs_values / dns_norm)

            output = Path(output)
            name = output.stem
            rve_results = output.parent / (name + f"_{cell_index}.xdmf")

            xdmf = ResultFile(rve_results)
            xdmf.add_function(u_rb, name="u_rb")
            xdmf.add_function(dns, name="dns")
            xdmf.add_function(aerr, name="aerr")
            xdmf.add_function(rerr, name="rerr")

            if output_strain:
                e = df.sym(df.grad(aerr))
                evoigt = df.as_vector([e[0, 0], e[1, 1], e[0, 1]])
                strain = df.Function(VDG)
                lp = LocalProjector(evoigt, VDG, df.dx)
                lp(strain)
                xdmf.add_function(strain, name="strain abs err")

            xdmf.write(0)
            xdmf.close()

        # dont forget to translate back to origin
        reference_domain.mesh.translate(df.Point(-offset))

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


def read_bases(*args, modes_per_edge=None):
    """read bases from args where each arg is a tuple of FilePath and
    tuple of string defining edge(s) for which to load basis functions from FilePath
    e.g. arg=("PathToFile", ("b", "r")) meaning
        load basis functions for bottom and right from "PathToFile"

    """
    bases = []
    edges = []
    for basis, edge_set in args:
        bases.append(np.load(basis))
        edges.append(list(edge_set))

    # check coarse scale basis phi and all edges are defined
    alle = [s for sub_set in edges for s in sub_set]
    assert not len(set(alle).difference(["phi", "b", "r", "t", "l"]))

    R = []
    Nmodes = []

    # append coarse scale basis functions
    for i, edge_set in enumerate(edges):
        if "phi" in edge_set:
            R.append(bases[i]["phi"])
            edge_set.remove("phi")

    # determine max number of modes per edge
    max_modes = []
    for basis, edge_set in zip(bases, edges):
        if edge_set:
            max_modes.append(max([len(basis[e]) for e in edge_set]))
    max_modes = max(max_modes)
    if modes_per_edge is not None:
        m = int(modes_per_edge)
    else:
        m = max_modes

    # append fine scale basis functions
    for key in ["b", "r", "t", "l"]:
        for basis, edge_set in zip(bases, edges):
            if key in edge_set:
                rb = basis[key][:m]
                Nmodes.append(rb.shape[0])
                if rb.shape[0] < m:
                    # add zero dummy modes (in case of dirichlet or neumann boundary)
                    # such that rb.shape[0] == m
                    rb = np.vstack((rb, np.zeros((m - rb.shape[0], rb.shape[1]))))
                R.append(rb)
                break

    return np.vstack(R), tuple(Nmodes)


# @deprecated
def read_basis(
    inner,
    inner_edges=("b", "r", "t", "l"),
    outer=None,
    outer_edges=(),
    modes_per_edge=None,
):
    """read multiscale basis

    Parameters
    ----------
    inner : str, Path
        FilePath to multiscale basis for domain Ω in the interior.
    inner_edges : tuple of str, optional
        The edges of the domain inside the global domain.
    outer : str, Path, optional
        FilePath to multiscale basis for domain Ω in case ∂Ω ∩ ∂Ω_gl is not empty.
    outer_edges : tuple of str, optional
        The edges of the domain Ω which are part of the boundary of the global domain.
    modes_per_edge : int, optional
        Number of basis functions per edge.

    Returns
    -------
    r : np.ndarray
        The basis.
    nmodes : tuple of int
        Maximum number of NON-ZERO modes on each edge.

    """
    r = []
    nmodes = []
    edges = ("b", "r", "t", "l")
    if set(edges).difference(set(inner_edges).union(outer_edges)):
        raise KeyError(f"A quadrilateral domain only has four edges {edges}.")

    # read basis
    inner_modes = np.load(inner)
    outer_modes = np.load(outer) if outer is not None else None

    # determine max number of modes
    max_inner = min([len(inner_modes[e]) for e in inner_edges])
    if outer_edges:
        max_outer = max([len(outer_modes[e]) for e in outer_edges])
    else:
        max_outer = 0
    max_modes = max([max_inner, max_outer])
    if modes_per_edge is not None and modes_per_edge < max_modes:
        m = int(modes_per_edge)
    else:
        m = max_modes

    # coarse scale basis is read from inner by default
    r.append(inner_modes["phi"])

    # fine scale modes
    for key in edges:
        if key in inner_edges:
            r.append(inner_modes[key][:m])
            nmodes.append(m)
        else:
            v = outer_modes[key][:m]
            nmodes.append(v.shape[0])
            if v.shape[0] < m:
                # add zero dummy modes s.t. v.shape[0] == m
                v = np.vstack((v, np.zeros((m - v.shape[0], v.shape[1]))))
            r.append(v)

    return np.vstack(r), tuple(nmodes)


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


def set_zero_at_dofs(U, dofs, atol=1e-6):
    """set U to zero at dofs if within relaxed tolerance"""
    zero = U.dofs(dofs)
    if np.allclose(np.zeros_like(zero), zero, atol=atol):
        # set respective dofs to zero
        # this modifies U._data in-place
        array = U.to_numpy()
        array[:, dofs] = np.zeros_like(zero)


def locate_dofs(x_dofs, X, gdim=2, s_=np.s_[:], tol=1e-9):
    """returns dofs at coordinates X

    Parameters
    ----------
    x_dofs : np.ndarray
        An array containing the coordinates of the DoFs of the FE space.
        Most likely the return value of V.tabulate_dof_coordinates().
    X : list, np.ndarray
        A list of points, where each point is given as list of len(gdim).
    gdim : int, optional
        The geometrical dimension of the domain.
    s_ : slice, optional
        Return slice of the dofs at each point.
    tol : float, optional
        Tolerance used to find coordinate.

    Returns
    -------
    dofs : np.ndarray
        DoFs at given coordinates.
    """
    if isinstance(X, list):
        X = np.array(X).reshape(len(X), gdim)
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        elif X.ndim > 2:
            raise NotImplementedError

    dofs = np.array([], int)
    for x in X:
        p = np.abs(x_dofs - x)
        v = np.where(np.all(p < tol, axis=1))[0]
        if v.size < 1:
            raise IndexError(f"The point {x} is not a vertex of the grid!")
        dofs = np.append(dofs, v[s_])

    return dofs
