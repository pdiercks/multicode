import pathlib
import numpy as np
import dolfin as df
from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator
from multi.io import ResultFile


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
    product_matrix = problem.discretize_product(product, bcs=False)
    product = FenicsMatrixOperator(product_matrix, V, V) if product_matrix else None

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

        xdmf = ResultFile(output.parent / (basename + f"_{cell_index}.xdmf"))
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
    product_matrix = problem.discretize_product(product, bcs=False)
    product = FenicsMatrixOperator(product_matrix, V, V) if product_matrix else None

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

    global_err_norm = np.sqrt(np.sum(err_norm ** 2))
    global_fom_norm = np.sqrt(np.sum(fom_norm ** 2))
    return {
        "err_norm": err_norm,
        "fom_norm": fom_norm,
        "rom_norm": rom_norm,
        "global_err_norm": global_err_norm,
        "global_fom_norm": global_fom_norm,
    }
