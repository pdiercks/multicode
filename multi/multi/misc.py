"""miscellaneous helpers"""

from pathlib import Path
import dolfin as df
import numpy as np
from multi import Domain, ResultFile, LinearElasticityProblem
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace


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


def compute_error_norm(
    rvemeshfile, degree, material, dofmap, udns, urb, basis, product=None, output=None
):
    """compute absolute and relative error (in norm given by product) of ROM solution wrt DNS

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

    Returns
    -------
    r : np.ndarray
        Global relative error.

    """

    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]

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
        if product:
            # FIXME compute only once, since matrix does not change in linear case
            p_mat = problem.get_product(name=product, bcs=False)
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
        norm["dns"].append(dns_norm)
        norm["abs_err"].append(err_norm)

        if output:
            aerr = df.Function(V)
            rerr = df.Function(V)
            abs_values = np.abs(dns.vector().get_local() - u_rb.vector().get_local())
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
            xdmf.write(0)
            xdmf.close()

    global_abs_err = np.sqrt(np.sum(np.array(norms["abs_err"]) ** 2))
    global_dns_norm = np.sqrt(np.sum(np.array(norms["dns_norm"]) ** 2))
    return global_abs_err / global_dns_norm
