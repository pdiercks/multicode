from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Union
import pathlib
import yaml
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

import ufl
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem, la, mesh, default_scalar_type
from dolfinx.fem.petsc import create_vector, create_matrix, set_bc, apply_lifting, assemble_vector, assemble_matrix
from dolfinx.fem.petsc import LinearProblem as LinearProblemBase
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from petsc4py import PETSc

from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator

from multi.bcs import BoundaryConditions
from multi.domain import Domain, StructuredQuadGrid, RectangularSubdomain
from multi.dofmap import QuadrilateralDofLayout
from multi.interpolation import make_mapping
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.sampling import create_random_values
from multi.utils import LogMixin


class LinearProblem(ABC, LinearProblemBase, LogMixin):
    """Class for solving a linear variational problem"""

    def __init__(self, domain: Domain, space: fem.FunctionSpaceBase):
        """Initialize domain and FE space

        Args:
            domain: The computational domain.
            space: The FE space.

        """
        self.domain = domain
        self.V = space
        self.trial = ufl.TrialFunction(self.V)
        self.test = ufl.TestFunction(self.V)
        self._bc_handler = BoundaryConditions(domain.grid, self.V, domain.facet_tags)

    def add_dirichlet_bc(
        self, value, boundary=None, sub=None, method="topological", entity_dim=None
    ):
        """see multi.bcs.BoundaryConditions.add_dirichletb_bc"""
        self._bc_handler.add_dirichlet_bc(
            value, boundary=boundary, sub=sub, method=method, entity_dim=entity_dim
        )

    def add_neumann_bc(self, marker, value):
        """see multi.bcs.BoundaryConditions.add_neumann_bc"""
        self._bc_handler.add_neumann_bc(marker, value)

    def clear_bcs(self, dirichlet=True, neumann=True):
        """remove all Dirichlet and Neumann bcs"""
        self._bc_handler.clear(dirichlet=dirichlet, neumann=neumann)

    def get_dirichlet_bcs(self):
        """The Dirichlet bcs"""
        # NOTE instance of dolfinx.fem.petsc.LinearProblem has attribute bcs
        return self._bc_handler.bcs

    @property
    @abstractmethod
    def form_lhs(self) -> ufl.Form:
        """The ufl form of the left hand side"""

    @property
    @abstractmethod
    def form_rhs(self) -> ufl.Form:
        """The ufl form of the right hand side"""

    def setup_solver(self, petsc_options: Optional[dict] = None,
                     form_compiler_options: Optional[dict] = None,
                     jit_options: Optional[dict] = None):
        """setup the solver for a linear variational problem

        This code is part of dolfinx.fem.petsc.py:
        Copyright (C) 2018-2022 Garth N. Wells and Jørgen S. Dokken

        Args:
            petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_options: Options used in FFCx compilation of
                this form. Run ``ffcx --help`` at the commandline to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See `python/dolfinx/jit.py` for
                all available options. Takes priority over all other
                option values.
        """

        a = self.form_lhs
        L = self.form_rhs
        bcs = self.get_dirichlet_bcs()

        # FIXME there are issues with PETSc.Options()
        # see here https://gitlab.com/petsc/petsc/-/issues/1201
        # Avoid using global PETSc.Options until the issues is closed.
        # super().__init__(
        #     a,
        #     L,
        #     bcs,
        #     petsc_options=petsc_options,
        #     form_compiler_options=form_compiler_options,
        #     jit_options=jit_options,
        # )

        # Get tpye aliases for "Form" and possibly other dolfinx types
        # simplified version of super().__init__ as workaround
        self._a = fem.form(a, form_compiler_options=form_compiler_options, jit_options=jit_options)
        self._A = create_matrix(self._a)

        self._L = fem.form(L, form_compiler_options=form_compiler_options, jit_options=jit_options)
        self._b = create_vector(self._L)

        # solution function
        self.u = fem.Function(self.V)

        self._x = la.create_petsc_vector_wrap(self.u.x)
        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        # set petsc options
        petsc_options = petsc_options or {}
        ksp_type = petsc_options.get("ksp_type", "preonly")
        pc_type = petsc_options.get("pc_type", "lu")
        pc_factor_mat_solver_type = petsc_options.get("pc_factor_mat_solver_type", "mumps")

        self._solver.setType(ksp_type)
        self._solver.getPC().setType(pc_type)
        self._solver.getPC().setFactorSolverType(pc_factor_mat_solver_type)

    def assemble_matrix(self, bcs=[]):
        """assemble matrix and apply boundary conditions"""
        self._A.zeroEntries()
        assemble_matrix(self._A, self._a, bcs=bcs)
        self._A.assemble()

    def assemble_vector(self, bcs=[]):
        """assemble vector and apply boundary conditions"""

        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, bcs)

    def __del__(self):
        # problem may be instantiated without call to `setup_solver`
        try:
            self._solver.destroy()
            self._A.destroy()
            self._b.destroy()
            self._x.destroy()
        except AttributeError:
            pass


class LinearElasticityProblem(LinearProblem):
    """Represents a linear elastic problem."""

    def __init__(
            self, domain: Domain, space: fem.FunctionSpaceBase, phases: Union[LinearElasticMaterial, list[tuple[LinearElasticMaterial, int]]]
    ):
        """Initializes a linear elastic problem.

        Args:
            domain: The computational domain.
            space: The FE space.
            phases: List of Tuple of linear elastic materials and cell tag for each phase.

        """

        super().__init__(domain, space)
        if isinstance(phases, LinearElasticMaterial):
            self.dx = ufl.dx
            self.phases = [phases]
        elif isinstance(phases, list):
            assert domain.cell_tags is not None
            self.dx = ufl.Measure("dx", domain=domain.grid,
                                  subdomain_data=domain.cell_tags)
            for (_, tag) in phases:
                assert tag in domain.cell_tags.values
            self.phases = phases
        self.gdim = domain.grid.ufl_cell().geometric_dimension()

    @property
    def form_lhs(self) -> ufl.Form:
        """get bilinear form a(u, v) of the problem"""
        u = self.trial
        v = self.test
        if len(self.phases) > 1:
            form = 0
            for (mat, i) in self.phases:
                form += ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx(i)
            return form
        else:
            mat = self.phases[0]
            return ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx

    # FIXME allow for body forces
    @property
    def form_rhs(self) -> ufl.Form:
        """get linear form f(v) of the problem"""
        domain = self.V.mesh
        v = self.test
        zero = fem.Constant(domain, (default_scalar_type(0.0),) * self.gdim)
        rhs = ufl.inner(zero, v) * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs

# FIXME should be an abstract base class; no direct use
class SubdomainProblem(object):
    """Represents a subproblem in a multiscale context"""


    def setup_edge_spaces(self) -> None:

        if not all([hasattr(self.domain, "fine_edge_grid"), hasattr(self.domain, "coarse_edge_grid")]):
            raise AttributeError("Fine and coarse grid partition of the edges does not exist.")

        # FIXME pyright complains: cannot access member "domain" etc
        edge_meshes = {"fine": self.domain.fine_edge_grid, "coarse": self.domain.coarse_edge_grid}
        V = self.V
        ufl_element = V.ufl_element()
        family = ufl_element.family_name
        shape = ufl_element.value_shape()

        edge_spaces = {}
        degree = {"coarse": 1, "fine": ufl_element.degree()}
        for scale, data in edge_meshes.items():
            edge_spaces[scale] = {}
            for edge, grid in data.items():
                fe = element(family, grid.basix_cell(), degree[scale], shape=shape, gdim=grid.ufl_cell().geometric_dimension())
                space = fem.functionspace(grid, fe)
                edge_spaces[scale][edge] = space

        self.edge_spaces = edge_spaces

    def setup_coarse_space(self) -> None:
        try:
            coarse_grid = self.domain.coarse_grid
        except AttributeError:
            raise AttributeError("Coarse grid partition of the domain does not exist.")
        V = self.V
        ufl_element = V.ufl_element()
        family_name = ufl_element.family_name
        shape = ufl_element.value_shape()
        fe = element(family_name, coarse_grid.basix_cell(), 1, shape=shape, gdim=coarse_grid.ufl_cell().geometric_dimension())
        self.W = fem.functionspace(coarse_grid, fe)

    def create_map_from_V_to_L(self) -> None:
        try:
            edge_spaces = self.edge_spaces
        except AttributeError:
            self.setup_edge_spaces()
            edge_spaces = self.edge_spaces

        V = self.V
        V_to_L = {}
        for edge, L in edge_spaces["fine"].items():
            V_to_L[edge] = make_mapping(L, V)
        self.V_to_L = V_to_L


class LinElaSubProblem(LinearElasticityProblem, SubdomainProblem):
    """Linear elasticity problem defined on a subdomain."""

    def __init__(self, domain: RectangularSubdomain, space: fem.FunctionSpaceBase, phases: tuple[LinearElasticMaterial]):
        super().__init__(domain, space, phases)



class TransferProblem(LogMixin):
    """Represents a Transfer Problem.

    This class aids the solution of transfer problems given by:

        A(u) = 0 in Ω,
        with optional homogeneous Dirichlet bcs on Γ_D,
        with arbitrary inhomogeneous Dirichlet boundary conditions on Γ_out.

    Here, we are only interested in the solution u restricted to the
    space defined on a target subdomain Ω_in ⊂ Ω.
    The boundaries Γ_D, Γ_N and Γ_N_inhom are the part of ∂Ω that
    intersects with the (respective dirichlet or neumann) boundary
    of the global domain Ω_gl ( Ω ⊂ Ω_gl).
    The boundary Γ_out is the part of ∂Ω that does not intersect with ∂Ω_gl.

    Note: The above problem can be formulated as a transfer operator which
    maps the boundary data on Γ_out (source space) to the solution u in
    the (range) space defined on Ω_in. Since FEniCS(x) does
    not allow to define function spaces on some part of the boundary
    of a domain (yet), the full space is defined as the source space.
    The range space is the space defined on Ω_in.

    Args:
        problem: The problem defined on the oversampling domain Ω.
        subproblem: The problem defined on the target subdomain Ω_in.
        gamma_out: Marker function that defines the boundary Γ_out geometrically.
        Note that `dolfinx.mesh.locate_entities_boundary` is used to determine the facets.
        dirichlet: Optional homogeneous Dirichlet BCs. See `multi.bcs.BoundaryConditions.add_dirichlet_bc` for suitable values.
        source_product: The inner product `Operator` to use for the source space
        or a `dict` that defines the inner product, see `multi.product.InnerProduct`.
        range_product: The inner product `Operator` to use for the range space
        or a `dict` that defines the inner product, see `multi.product.InnerProduct`.
        kernel: The kernel (rigid body modes) of A. If not None, the part of the
        solution orthogonal to this kernel is returned by `solve`.

    Note:
        The kernel needs to be orthonormal wrt chosen range product.

    """

    def __init__(
        self,
        problem: LinearElasticityProblem,
        subproblem: LinElaSubProblem,
        gamma_out: Callable,
        dirichlet: Optional[Union[dict, list[dict]]] = None,
        source_product: Optional[Union[Operator, dict]] = None,
        range_product: Optional[Union[Operator, dict]] = None,
        kernel: Optional[VectorArray] = None,
    ):
        self.problem = problem
        self.subproblem = subproblem
        self.source = FenicsxVectorSpace(problem.V)
        self.range = FenicsxVectorSpace(subproblem.V)
        self.gamma_out = gamma_out
        self.kernel = kernel

        # initialize commonly used quantities
        self._init_bc_gamma_out()
        self._S_to_R = self._make_mapping()

        # initialize fixed set of dirichlet boundary conditions on Γ_D
        self._bc_hom = list()
        if dirichlet is not None:
            if isinstance(dirichlet, (list, tuple)):
                for dirichlet_bc in dirichlet:
                    problem.add_dirichlet_bc(**dirichlet_bc)
            else:
                problem.add_dirichlet_bc(**dirichlet)
            self._bc_hom = problem.get_dirichlet_bcs()

        # ### inner products
        default = {'product': 'euclidean', 'bcs': ()}
        source_prod = source_product or default
        range_prod = range_product or default
        self.source_product = self._init_source_product(source_prod)
        self.range_product = self._init_range_product(range_prod)

    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.source.V
        tdim = V.mesh.topology.dim
        fdim = tdim - 1
        dummy = fem.Function(V)

        # determine boundary facets of Γ_out
        facets_Γ_out = mesh.locate_entities_boundary(
            V.mesh, fdim, self.gamma_out
        )
        # determine dofs on Γ_out
        _dofs = fem.locate_dofs_topological(V, fdim, facets_Γ_out)
        bc = fem.dirichletbc(dummy, _dofs)

        # only internal use
        self._facets_Γ_out = facets_Γ_out
        self._fdim = fdim
        self._dummy_bc_gamma_out = bc
        self._dofs_Γ_out = _dofs

        # quantities that might be used --> property
        dofs = bc._cpp_object.dof_indices()[0]
        self._bc_dofs_gamma_out = dofs
        # source space restricted to Γ_out
        self._source_gamma = NumpyVectorSpace(dofs.size)

    @property
    def bc_hom(self):
        """homogeneous Dirichlet bcs on Γ_D"""
        return self._bc_hom

    @property
    def bc_dofs_gamma_out(self):
        """dof indices associated with Γ_out"""
        return self._bc_dofs_gamma_out

    @property
    def source_gamma_out(self):
        """NumpyVectorSpace of dim `self.bc_dofs_gamma_out.size`"""
        return self._source_gamma

    @property
    def S_to_R(self):
        """map from source to range space"""
        return self._S_to_R

    def _make_mapping(self):
        """builds map from source space to range space"""
        return make_mapping(self.range.V, self.source.V)

    def discretize_operator(self):
        """discretize the operator A of the oversampling problem"""
        V = self.source.V
        Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
        self.logger.debug(f"Discretizing left hand side of the problem (size={Vdim}).")

        p = self.problem
        # need to add u=g on Γ_D and u=r on Γ_out such that
        # rows and columns are correctly modified
        bc_hom = self.bc_hom
        bc_inhom = self._dummy_bc_gamma_out
        bcs = [bc_inhom] + bc_hom

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        p.setup_solver(petsc_options=petsc_options)
        p.assemble_matrix(bcs=bcs)
        # now p.solver is setup with matrix p.A
        # and p.a should be used to modify rhs (apply lifting)

    def discretize_rhs(self, boundary_values):
        """discretize the right hand side

        Parameters
        ----------
        boundary_values : np.ndarray
            The values to prescribe on Γ_out.

        """

        # boundary data defines values on Γ_out
        # zero values on Γ_D if present are added here without the user
        # having to care about this

        dofs = self.bc_dofs_gamma_out
        _dofs = self._dofs_Γ_out
        p = self.problem
        f = fem.Function(p.V)
        f.x.array[dofs] = boundary_values
        bc_inhom = fem.dirichletbc(f, _dofs)

        p.assemble_vector(bcs=[bc_inhom])

    def generate_random_boundary_data(
            self, count: int, distribution: str, options: Optional[dict[str, Any]] = None
    ) -> npt.NDArray:
        """Generates random vectors of shape (count, num_dofs_Γ_out).

        Args:
            count: Number of random vectors.
            distribution: The distribution used for sampling.
            options: Arguments passed to sampling method of random number generator.

        """

        bc_dofs = self.bc_dofs_gamma_out  # actual size of the source space
        options = options or {}
        values = create_random_values(
            (count, bc_dofs.size), distribution, **options
        )

        return values

    def solve(self, boundary_values):
        """solve the problem for boundary_data

        Parameters
        ----------
        boundary_values : np.ndarray
            The values to prescribe on Γ_out.

        Returns
        -------
        U_in : VectorArray
            The solutions in the range space.
        """


        self.logger.info(f"Solving TransferProblem for {len(boundary_values)} vectors.")

        p = self.problem
        try:
            solver = p.solver
        except AttributeError:
            self.discretize_operator()
            solver = p.solver

        rhs = p.b

        # solution
        u = fem.Function(p.V)  # full space
        u_in = fem.Function(self.range.V)  # target subdomain
        U = self.range.empty()  # VectorArray to store u_in

        # construct rhs from boundary data
        for array in boundary_values:
            self.discretize_rhs(array)
            solver.solve(rhs, u.vector)
            u.x.scatter_forward()

            # ### restrict full solution to target subdomain
            # need to use nmm_interpolation_data even if subdomain discretization
            # matches the global mesh
            u_in.interpolate(u, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                u_in.function_space.mesh._cpp_object,
                u_in.function_space.element,
                u.function_space.mesh._cpp_object))
            U.append(self.range.make_array([u_in.vector.copy()]))

        if self.kernel is not None:
            assert len(self.kernel) > 0
            return orthogonal_part(U, self.kernel, product=self.range_product, orthonormal=True)
        else:
            return U

    def _init_source_product(self, product: Union[Operator, dict]) -> Union[Operator, None]:
        """Initializes the source product."""
        if isinstance(product, Operator):
            return product
        else:
            product_name = product['product']
            bcs = product['bcs']
            inner_product = InnerProduct(self.problem.V, product_name, bcs=bcs)
            matrix = inner_product.assemble_matrix()
            if matrix is None:
                return None
            else:
                full_matrix = csr_array(matrix.getValuesCSR()[::-1])
                dofs = self.bc_dofs_gamma_out
                source_matrix = full_matrix[dofs, :][:, dofs]
                source_product = NumpyMatrixOperator(source_matrix)
                return source_product

    def _init_range_product(self, product: Union[Operator, dict]) -> Union[Operator, None]:
        """Initializes the range product."""
        if isinstance(product, Operator):
            return product
        else:
            product_name = product['product']
            bcs = product['bcs']
            range_product = InnerProduct(self.subproblem.V, product_name, bcs=bcs)
            matrix = range_product.assemble_matrix()
            if matrix is None:
                return None
            else:
                return FenicsxMatrixOperator(matrix, self.subproblem.V, self.subproblem.V)


class MultiscaleProblemDefinition(ABC):
    """Base class to define a multiscale problem."""

    def __init__(self, coarse_grid: str, fine_grid: str) -> None:
        """Initializes a multiscale problem.

        Args:
            coarse_grid: The coarse scale discretization of the global domain.
            fine_grid: The fine scale discretization of the global domain.
        """
        self.coarse_grid_path = pathlib.Path(coarse_grid)
        self.fine_grid_path = pathlib.Path(fine_grid)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, yaml_file: str):
        with open(yaml_file, "r") as instream:
            mat = yaml.safe_load(instream)
        self._material = mat

    def setup_coarse_grid(self, gdim: int):
        """Reads the coarse scale grid from file.

        Args:
            gdim: Geometric dimension.

        """
        domain, ct, ft = gmshio.read_from_msh(
            self.coarse_grid_path.as_posix(), MPI.COMM_SELF, gdim=gdim
        )
        self.coarse_grid = StructuredQuadGrid(domain, ct, ft)

    def setup_fine_grid(self, cell_tags: bool = False) -> None:
        """Reads the fine scale grid from file.

        Args:
            cell_tags: If True, read meshtags from XDMFFile.

        Note:
            If `self.boundaries` is not None, facet tags are
            created accordingly.
        """
        fine_ct = None
        with XDMFFile(
            MPI.COMM_SELF, self.fine_grid_path.as_posix(), "r"
        ) as xdmf:
            fine_domain = xdmf.read_mesh(name="Grid")
            if cell_tags:
                fine_ct = xdmf.read_meshtags(fine_domain, name="Grid")

        fine_ft = None
        boundaries = self.boundaries
        if boundaries is not None:
            from multi.preprocessing import create_meshtags
            tdim = fine_domain.topology.dim
            fdim = tdim - 1
            fine_ft, _ = create_meshtags(fine_domain, fdim, boundaries)
        self.fine_grid = Domain(
            fine_domain, cell_tags=fine_ct, facet_tags=fine_ft
        )

    def setup_coarse_space(self, family: str = "P", degree: int = 1, shape: tuple[int, ...] = (2,)) -> None:
        """Creates FE space on coarse grid.

        Args:
            family: The FE family.
            degree: The interpolation order.
            shape: The value shape.

        """
        grid = self.coarse_grid.grid
        fe = element(family, grid.basix_cell(), degree, shape=shape)
        self.W = fem.functionspace(grid, fe)

    def setup_fine_space(self, family: str = "P", degree: int = 2, shape: tuple[int, ...] = (2,)) -> None:
        """Creates FE space on fine grid.

        Args:
            family: The FE family.
            degree: The interpolation order.
            shape: The value shape.

        """
        grid = self.fine_grid.grid
        fe = element(family, grid.basix_cell(), degree, shape=shape)
        self.V = fem.functionspace(grid, fe)

    @property
    @abstractmethod
    def cell_sets(self) -> dict[str, set[int]]:
        """Cell sets for the particular problem."""
        pass

    @property
    @abstractmethod
    def boundaries(self) -> dict[str, tuple[int, Callable]]:
        """Boundaries of the global domain.

        Returns:
            boundaries: The boundary names are given as keys and each value
            defines the boundary id and a geometrical marker function.
        """
        pass

    @abstractmethod
    def get_dirichlet(self, cell_index: Optional[int] = None) -> Union[dict, None]:
        """Returns Dirichlet BCs.

        Args:
            cell_index: Optional cell index.

        Note:
            If cell_index is None, returns Dirichlet BCs for global problem.
            Otherwise Dirichlet BCs only relevant to given cell are returned.
            This can be used to define BCs of oversampling problems.
        """
        pass

    @abstractmethod
    def get_neumann(self, cell_index: Optional[int] = None) -> Union[dict, None]:
        """Returns Neumann BCs.

        Args:
            cell_index: Optional cell index.

        Note:
            If cell_index is None, returns Neumann BCs for global problem.
            Otherwise Neumann BCs only relevant to given cell are returned.
            This can be used to define BCs of oversampling problems.
        """
        pass

    @abstractmethod
    def get_gamma_out(self, cell_index: int) -> Callable:
        """Returns boundary Γ_out of oversampling problem.

        Args:
            cell_index: The global cell index of the target subdomain.
        """
        pass

    @abstractmethod
    def get_kernel_set(self, cell_index: int) -> tuple[int, ...]:
        """Returns set of vectors spanning nullspace of oversampling problem.

        Args:
            cell_index: The global cell index of the target subdomain.
        """
        pass

    def build_edge_basis_config(self, cell_sets: dict[str, set[int]]) -> None:
        """Builds mappings from cell index to _active edges_ and from edge
        index to cell containing that edge. _Active_ refers to whether the cell
        owns this edge or if it is owned by the neighbouring coarse grid cell.
        This information is used to compute the extension of fine scale edge
        basis functions to achieve a conforming global approximation.
        See also ``self.active_edges`` and ``self.edge_to_cell``.

        Args:
            cell_sets: Cell sets used to define _active edges_. The order
            is important.

        Example:
            The edge e is shared by cells A (right) and B (left). Providing the cell sets
            {"A": set([A]), "B": set([B])} in this order will result in A
            owning the edge e. Thus, the fine scale edge modes generated for the right
            edge of A will also be extended into coarse grid cell B with the modes
            prescribed on B's left edge.

        """

        cs = {}
        # sort cell indices in increasing order
        for key, value in cell_sets.items():
            cs[key] = np.sort(list(value))

        dof_layout = QuadrilateralDofLayout()
        active_edges = {}
        edge_map = {}  # maps global edge index to tuple(cell index, local edge)
        marked_edges = set()

        for cset in cs.values():
            for cell_index in cset:

                active_edges[cell_index] = set()
                edges = self.coarse_grid.get_entities(1, cell_index)
                for local_ent, ent in enumerate(edges):
                    edge = dof_layout.local_edge_index_map[local_ent]
                    if ent not in marked_edges:
                        active_edges[cell_index].add(edge)
                        edge_map[ent] = (cell_index, edge)
                        marked_edges.add(ent)
        assert len(active_edges.keys()) == self.coarse_grid.num_cells
        self._active_edges = active_edges
        self._edge_map = edge_map

    def active_edges(self, cell: int) -> set[str]:
        """Returns the set of local edges owned by given cell.

        Args:
            cell_index: The global cell index of the target subdomain.

        """
        if not hasattr(self, "_active_edges"):
            raise AttributeError(
                "You have to define an edge basis configuration "
                "by calling `self.build_edge_basis_config`"
            )
        return self._active_edges[cell]

    def edge_to_cell(self, edge: int) -> tuple[int, str]:
        """Returns cell owning the edge and str representation of local edge
        entity.

        Args:
            edge: The global edge index.

        """
        if not hasattr(self, "_edge_map"):
            raise AttributeError(
                "You have to define an edge basis configuration "
                "by calling `self.build_edge_basis_config`"
            )
        return self._edge_map[edge]
