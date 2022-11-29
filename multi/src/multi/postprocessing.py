import pathlib
import yaml
import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx.io import XDMFFile
import matplotlib.pyplot as plt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator

from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct


def plot_modes(edge_space, edge, modes, component, mask):
    """for hacking"""
    L = edge_space
    x_dofs = L.sub(0).collapse().tabulate_dof_coordinates()

    if component in ("x", 0):
        modes = modes[:, ::2]
    elif component in ("y", 1):
        modes = modes[:, 1::2]

    if edge in ("b", "t"):
        xx = x_dofs[:, 0]
    elif edge in ("r", "l"):
        xx = x_dofs[:, 1]
    oo = np.argsort(xx)

    for mode in modes[mask]:
        plt.plot(xx[oo], mode[oo])
    plt.show()


# def visualize_edge_modes(
#     meshfile, basispath, edge, component, degree=2, N=4, fid=1, show=False
# ):
#     data = np.load(basispath)
#     modes = data[edge]
#     edge_domain = Domain(meshfile)
#     L = df.VectorFunctionSpace(edge_domain.mesh, "CG", degree)
#     x_dofs = L.sub(0).collapse().tabulate_dof_coordinates()

#     if component in ("x", 0):
#         modes = modes[:, ::2]
#     elif component in ("y", 1):
#         modes = modes[:, 1::2]

#     if edge in ("b", "t"):
#         xx = x_dofs[:, 0]
#     elif edge in ("r", "l"):
#         xx = x_dofs[:, 1]
#     oo = np.argsort(xx)

#     plt.figure(fid)
#     for mode in modes[:N]:
#         plt.plot(xx[oo], mode[oo])
#     if show:
#         plt.show()


def read_bam_colors():
    """get primary(0), secondary(1), tertiary(2) or all(3) for
    bam colors black, blue, green red, and yellow"""
    yamlfile = pathlib.Path(__file__).parent / "bamcolors_rgb.yml"
    with yamlfile.open("r") as instream:
        bam_cd = yaml.safe_load(instream)
    return bam_cd


def read_bam_colormap():
    infile = pathlib.Path(__file__).parent / "bam-RdBu.npy"
    cmap = np.load(infile.as_posix())
    return cmap


def write_local_fields(xdmf_file, multiscale_problem, dofmap, bases, u_rom, u_fom):
    """get local field for each subdomain and write to file"""

    output = pathlib.Path(xdmf_file)
    basename = output.stem

    degree = multiscale_problem.degree
    grid_dir = multiscale_problem.grid_dir

    num_cells = multiscale_problem.grid.num_cells
    for cell_index in range(num_cells):

        subdomain_xdmf = grid_dir / f"offline/subdomain_{cell_index:03}.xdmf"

        with XDMFFile(MPI.COMM_WORLD, subdomain_xdmf.as_posix(), "r") as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")
        V = dolfinx.fem.VectorFunctionSpace(subdomain, ("P", degree))

        # ### fom solution for cell
        u_fom_local = u_fom.interpolate(V)
        u_fom_local.name = "u_fom"

        # ### rom solution for cell
        dofs = dofmap.cell_dofs(cell_index)
        basis = bases[cell_index]
        u_rom = dolfinx.fem.Function(V, name="u_rom")
        u_rom_vec = u_rom.vector
        u_rom_vec.array[:] = basis.T @ u_rom[dofs]

        # ### absolute error
        aerr = dolfinx.fem.Function(V)
        aerr.name = "u_err"
        aerr_vec = aerr.vector
        aerr_vec.axpy(1.0, u_fom_local.vector)
        aerr_vec.axpy(-1.0, u_rom_vec)

        target = output.parent / (basename + f"_{cell_index:03}.xdmf")
        with XDMFFile(subdomain.comm, target.as_posix(), "w") as xdmf:
            xdmf.write_mesh(subdomain)
            xdmf.write_function(u_fom)
            xdmf.write_function(u_rom)


def compute_error_norms(
    multiscale_problem, dofmap, bases, u_rom, u_fom, product=None, xdmf_file=None
):
    """compute error norms

    Parameters
    ----------
    multiscale_problem : multi.problems.MultiscaleProblem
        The class defining the multiscale problem.
    dofmap : multi.dofmap.DofMap
        The dofmap of the multiscale problem.
    bases : list of np.ndarray
        The local reduced basis for each coarse grid cell.
    u_rom : np.ndarray
        The ROM solution.
    u_fom : dolfinx.fem.Function
        The FOM soultion (global fine grid space).
    product : optional, str
        The inner product wrt which the error is computed.
    xdmf_file : optional, str
        Write the local fields to XDMFFile.

    Returns
    -------
    norms : dict
        The (local) norm of the absolute error ('err'), the FOM solution ('fom')
        and the ROM solution ('rom') as key value pairs.
        Values are `np.ndarray`s.
    """

    degree = multiscale_problem.degree
    grid_dir = multiscale_problem.grid_dir

    # NOTE issue can be resolved by deleting the cash
    # as a consequence, I need to be careful with creating
    # the same function several times?

    num_cells = dofmap.grid.num_cells
    for cell_index in range(num_cells):

        subdomain_xdmf = grid_dir / f"offline/subdomain_{cell_index:03}.xdmf"

        with XDMFFile(MPI.COMM_WORLD, subdomain_xdmf.as_posix(), "r") as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")
        V = dolfinx.fem.VectorFunctionSpace(subdomain, ("P", degree))

        # ### fom solution for cell
        u_fom_local = dolfinx.fem.Function(V)
        u_fom_local.interpolate(u_fom)

        # ### rom solution for cell
        dofs = dofmap.cell_dofs(cell_index)
        basis = bases[cell_index]
        u_rom_local = dolfinx.fem.Function(V)
        u_rom_vec = u_rom_local.vector
        u_rom_vec.array[:] = basis.T @ u_rom[dofs]

        if product == "energy":
            # instantiate linear elasticity problem
            material = multiscale_problem.material
            E = material["Material parameters"]["E"]["value"]
            NU = material["Material parameters"]["NU"]["value"]
            plane_stress = material["Constraints"]["plane_stress"]

            cell_index = 0
            subdomain_xdmf = grid_dir / f"offline/subdomain_{cell_index:03}.xdmf"
            with XDMFFile(MPI.COMM_WORLD, subdomain_xdmf.as_posix(), "r") as xdmf:
                subdomain = xdmf.read_mesh(name="Grid")
                ct = xdmf.read_meshtags(subdomain, name="Grid")

            Ω = RectangularDomain(subdomain, cell_markers=ct)
            V = dolfinx.fem.VectorFunctionSpace(subdomain, ("P", degree))
            problem = LinearElasticityProblem(
                Ω, V, E=E, NU=NU, plane_stress=plane_stress
            )
            product = problem.get_form_lhs()

        inner_product = InnerProduct(V, product, bcs=())
        matrix = inner_product.assemble_matrix()
        if matrix is not None:
            product_op = FenicsxMatrixOperator(matrix, V, V)
        else:
            product_op = None

        # ### compute error
        source = FenicsxVectorSpace(V)
        fom = source.make_array([u_fom_local.vector])
        rom = source.make_array([u_rom_vec])
        err = fom - rom

        err_norm = err.norm(product_op)
        fom_norm = fom.norm(product_op)
        rom_norm = rom.norm(product_op)

        breakpoint()

    return {
        "err": err_norm,
        "fom": fom_norm,
        "rom": rom_norm,
    }
