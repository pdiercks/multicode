import pathlib
import yaml
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from pymor.bindings.fenics import FenicsVectorSpace
from multi.domain import Domain
from multi.io import ResultFile
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


def visualize_edge_modes(
    meshfile, basispath, edge, component, degree=2, N=4, fid=1, show=False
):
    data = np.load(basispath)
    modes = data[edge]
    edge_domain = Domain(meshfile)
    L = df.VectorFunctionSpace(edge_domain.mesh, "CG", degree)
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

    plt.figure(fid)
    for mode in modes[:N]:
        plt.plot(xx[oo], mode[oo])
    if show:
        plt.show()


def read_bam_colors():
    """get primary(0), secondary(1), tertiary(2) or all(3) for
    bam colors black, blue, green red, and yellow"""
    yamlfile = pathlib.Path(__file__).parent / "bamcolors_rgb.yml"
    with yamlfile.open("r") as instream:
        bam_cd = yaml.safe_load(instream)
    return bam_cd


def write_local_fields(
    xdmf_file, problem, dofmap, bases, cell_to_basis, u_rom, u_fom, product=None
):
    """get local field for each subdomain and write to file"""

    output = pathlib.Path(xdmf_file)
    basename = output.stem

    num_cells = len(dofmap.cells)
    assert cell_to_basis.shape[0] == num_cells
    assert cell_to_basis.max() < len(bases)
    V = problem.V

    source = FenicsVectorSpace(V)
    inner_product = InnerProduct(V, product, bcs=())
    product = inner_product.assemble_operator()

    for cell_index, cell in enumerate(dofmap.cells):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        problem.domain.translate(df.Point(offset))

        # ### fom solution for cell
        u_fom_local = df.interpolate(u_fom, V)

        # ### rom solution for cell
        u_rom_local = df.Function(V)
        u_rom_vector = u_rom_local.vector()
        dofs = dofmap.cell_dofs(cell_index)
        basis = bases[cell_to_basis[cell_index]]
        u_rom_vector.set_local(basis.T @ u_rom[dofs])

        # ### absolute error
        aerr = df.Function(V)
        aerr_vec = aerr.vector()
        aerr_vec.axpy(1.0, u_fom_local.vector())
        aerr_vec.axpy(-1.0, u_rom_vector)

        # ### relative error
        U_fom = source.make_array([u_fom_local.vector()])
        fom_norm = U_fom.norm(product)
        rerr = df.Function(V)
        rerr_vec = rerr.vector()
        rerr_vec.axpy(1.0, aerr_vec)
        rerr_vec /= fom_norm

        xdmf = ResultFile(output.parent / (basename + f"_{cell_index:03}.xdmf"))
        xdmf.add_function(u_rom_local, name="u-rom")
        xdmf.add_function(u_fom_local, name="u-fom")
        xdmf.add_function(aerr, name="aerr")
        xdmf.add_function(rerr, name="rerr")
        xdmf.write(0)
        xdmf.close()

        # translate back
        problem.domain.translate(df.Point(-offset))


def compute_local_error_norm(
    problem, dofmap, bases, cell_to_basis, u_rom, u_fom, product=None
):
    """compute local error norm"""

    num_cells = len(dofmap.cells)
    assert cell_to_basis.shape[0] == num_cells
    assert cell_to_basis.max() < len(bases)

    V = problem.V
    source = FenicsVectorSpace(V)
    inner_product = InnerProduct(V, product, bcs=())
    product = inner_product.assemble_operator()

    reconstructed = []
    fom_solutions = []
    u_rom_local = df.Function(V)
    u_rom_vector = u_rom_local.vector()
    for cell_index, cell in enumerate(dofmap.cells):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        problem.domain.translate(df.Point(offset))

        # ### fom solution for cell
        u_fom_local = df.interpolate(u_fom, V)
        fom_solutions.append(u_fom_local.vector())

        # ### rom for cell
        dofs = dofmap.cell_dofs(cell_index)
        basis = bases[cell_to_basis[cell_index]]
        u_rom_vector.set_local(basis.T @ u_rom[dofs])
        reconstructed.append(u_rom_vector.copy())

        # translate back
        problem.domain.translate(df.Point(-offset))

    # ### compute error
    fom = source.make_array(fom_solutions)
    rom = source.make_array(reconstructed)
    err = fom - rom

    err_norm = err.norm(product)
    fom_norm = fom.norm(product)
    rom_norm = rom.norm(product)

    global_err_norm = np.sqrt(np.sum(err_norm**2))
    global_fom_norm = np.sqrt(np.sum(fom_norm**2))
    return {
        "err_norm": err_norm,
        "fom_norm": fom_norm,
        "rom_norm": rom_norm,
        "global_err_norm": global_err_norm,
        "global_fom_norm": global_fom_norm,
    }
