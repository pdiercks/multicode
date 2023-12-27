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
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator

from multi.bcs import BoundaryConditions
from multi.domain import StructuredQuadGrid, Domain, RectangularSubdomain
from multi.dofmap import QuadrilateralDofLayout
from multi.interpolation import make_mapping
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.sampling import create_random_values
from multi.solver import build_nullspace
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

    def setup_solver(self, petsc_options={}, form_compiler_options={}, jit_options={}):
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
            self, domain: Domain, space: fem.FunctionSpaceBase, phases: tuple[LinearElasticMaterial, ...]
    ):
        """Initializes a linear elastic problem.

        Args:
            domain: The computational domain.
            space: The FE space.
            phases: Tuple of linear elastic materials for each phase.
            The order should match `domain.cell_tags` if there are
            several phases.

        """

        super().__init__(domain, space)
        if domain.cell_tags is None:
            assert len(phases) == 1
            self.dx = ufl.dx
        else:
            self.dx = ufl.Measure("dx", domain=domain.grid,
                                  subdomain_data=domain.cell_tags)
        self.gdim = domain.grid.ufl_cell().geometric_dimension()
        self.phases = phases

    @property
    def form_lhs(self) -> ufl.Form:
        """get bilinear form a(u, v) of the problem"""
        u = self.trial
        v = self.test
        if len(self.phases) > 1:
            form = 0
            for (i, mat) in enumerate(self.phases):
                form += ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx(i + 1)
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
        self.W = fem.VectorFunctionSpace(coarse_grid, (family_name, 1))

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
        source_product: The inner product to use for the source space. See `multi.product.InnerProduct`.
        range_product: The inner product to use for the range space. See `multi.product.InnerProduct`.
        remove_kernel: If True, remove kernel (rigid body modes) from solution.

    """

    def __init__(
        self,
        problem: LinearElasticityProblem,
        subproblem: LinElaSubProblem,
        gamma_out: Callable,
        dirichlet: Optional[Union[dict, list[dict]]] = None,
        source_product: Optional[dict] = None,
        range_product: Optional[dict] = None,
        remove_kernel: Optional[bool] = False,
    ):
        self.problem = problem
        self.subproblem = subproblem
        self.source = FenicsxVectorSpace(problem.V)
        self.range = FenicsxVectorSpace(subproblem.V)
        self.gamma_out = gamma_out
        self.remove_kernel = remove_kernel

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
        default_product = {"product": None, "bcs": (), "product_name": None}
        source_prod = source_product or default_product
        range_prod = range_product or default_product
        self.source_product = self._get_source_product(**source_prod)
        self.range_product = self._get_range_product(**range_prod)

        # initialize kernel
        if remove_kernel:
            # FIXME using self.range_product instead of "l2" as default would be more consistent?
            # build null space
            l2_product = self._get_range_product(product="l2")
            self.range_l2_product = l2_product
            self.kernel = build_nullspace(self.range, product=l2_product, gdim=2)

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

        if self.remove_kernel:
            return orthogonal_part(self.kernel, U, self.range_l2_product, orth=True)
        else:
            return U

    def _get_source_product(self, product=None, bcs=(), product_name=None):
        """get source product

        Parameters
        ----------
        product : str, optional
            The inner product to use.
        bcs : list of df.DirichletBC, optional
            The bcs to be applied to the product matrix.
        product_name : str, optional
            Name of the NumpyMatrixOperator.

        Returns
        -------
        source_product : NumpyMatrixOperator or None
        """
        if product is not None:
            inner_product = InnerProduct(
                self.problem.V, product, bcs=bcs, name=product_name
            )
            M = inner_product.assemble_matrix()
            full_matrix = csr_array(M.getValuesCSR()[::-1])
            dofs = self.bc_dofs_gamma_out
            source_matrix = full_matrix[dofs, :][:, dofs]
            source_product = NumpyMatrixOperator(source_matrix, name=product_name)
            return source_product
        else:
            return None

    def _get_range_product(self, product=None, bcs=(), product_name=None):
        """discretize range product

        Parameters
        ----------
        product : str, optional
            The inner product to use.
        bcs : list of df.DirichletBC, optional
            The bcs to be applied to the product matrix.
        product_name : str, optional
            Name of the FenicsMatrixOperator.

        Returns
        -------
        range_product : FenicsMatrixOperator or None
        """
        range_product = InnerProduct(self.range.V, product, bcs=bcs, name=product_name)
        matrix = range_product.assemble_matrix()
        if product is None:
            return None
        else:
            return FenicsxMatrixOperator(matrix, self.range.V, self.range.V)


# FIXME should be an ABC class
class MultiscaleProblemDefinition(object):
    """Base class to define a multiscale problem."""

    def __init__(self, coarse_grid_path, fine_grid_path):
        self.coarse_grid_path = pathlib.Path(coarse_grid_path)
        self.fine_grid_path = pathlib.Path(fine_grid_path)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, yaml_file):
        with open(yaml_file, "r") as instream:
            mat = yaml.safe_load(instream)
        self._material = mat

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        self._degree = int(degree)

    def setup_coarse_grid(self):
        """create coarse grid"""
        domain, ct, ft = gmshio.read_from_msh(
            self.coarse_grid_path.as_posix(), MPI.COMM_SELF, gdim=2
        )
        self.coarse_grid = StructuredQuadGrid(domain, ct, ft)

    def setup_fine_grid(self):
        """create fine grid"""
        with XDMFFile(
            MPI.COMM_SELF, self.fine_grid_path.as_posix(), "r"
        ) as xdmf:
            fine_domain = xdmf.read_mesh(name="Grid")
            fine_ct = xdmf.read_meshtags(fine_domain, name="Grid")

        boundaries = self.boundaries
        if boundaries is not None:
            from multi.preprocessing import create_facet_tags

            fine_ft, _ = create_facet_tags(fine_domain, boundaries)
        else:
            fine_ft = None
        self.fine_grid = Domain(
            fine_domain, cell_tags=fine_ct, facet_tags=fine_ft
        )

    def setup_coarse_space(self, family="P", degree=1, shape=(2,)):
        """create FE space on coarse grid"""
        grid = self.coarse_grid.grid
        fe = element(family, grid.basix_cell(), degree, shape=shape)
        self.W = fem.functionspace(grid, fe)

    def setup_fine_space(self, family="P", shape=(2,)):
        """create FE space on fine grid"""
        try:
            degree = self.degree
        except AttributeError:
            raise RuntimeError("You need to set the degree of the problem first")
        grid = self.fine_grid.grid
        fe = element(family, grid.basix_cell(), degree, shape=shape)
        self.V = fem.functionspace(grid, fe)

    @property
    def cell_sets(self):
        raise NotImplementedError

    @property
    def boundaries(self):
        raise NotImplementedError

    def get_dirichlet(self, cell_index=None):
        raise NotImplementedError

    def get_neumann(self, cell_index=None):
        raise NotImplementedError

    def get_gamma_out(self, cell_index):
        raise NotImplementedError

    def get_remove_kernel(self, cell_index):
        raise NotImplementedError

    def build_edge_basis_config(self, cell_sets):
        """defines which oversampling problem is used to
        compute the POD basis for a certain edge

        Parameters
        ----------
        cell_sets : dict
            Cell sets according to which 'active_edges' for
            a cell are defined. The order is important.

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
        self.active_edges = active_edges
        self.edge_map = edge_map

    def get_active_edges(self, cell_index):
        """returns the set of edges to consider for
        construction of POD basis in the TransferProblem for given cell.
        This depends on the `self.edge_basis_config`.

        """
        if not hasattr(self, "active_edges"):
            raise AttributeError(
                "You have to define an edge basis configuration "
                "by calling `self.edge_basis_config`"
            )
        config = self.active_edges
        return config[cell_index]
