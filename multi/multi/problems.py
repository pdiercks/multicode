import pathlib
import yaml
import dolfinx
from dolfinx.io import gmshio
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from multi.bcs import BoundaryConditions
from multi.domain import StructuredQuadGrid, Domain
from multi.dofmap import QuadrilateralDofLayout
from multi.interpolation import make_mapping
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.sampling import _create_random_values
from multi.solver import build_nullspace

from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.tools.random import get_random_state
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.timing import Timer

from scipy.sparse import csc_matrix


"""
Design

# 1. init
p = LinearProblem(domain, V)

# 2. set bcs unique to this problem
p.add_dirichlet_bc(...)
p.add_neumann_bc(...)

# make sure that p.get_form_lhs() and p.get_form_rhs() are implemented ...

# 3. setup solver
p.setup_solver(petsc_options, form_compiler_options, jit_options)

# ----------------
# now with the solver setup there are different use cases
# (A) solve the problem once and be done for today
p.solve() # will call super.solve()

# ----------------
# (B) we want to solve the same problem many times with different rhs
solver = p.solver # first get the solver we did setup
p.assemble_matrix(bcs) # assemble matrix once for specific set of bcs

# Note that p.A is filled with values internally
# Next, we only need to define the rhs for which we want to solve
# One option is to assemble the vector based on p.get_form_rhs()

p.assemble_vector(bcs) # let assemble vector always modify p.b
rhs = p.b
solution = dolfinx.fem.Function(p.V)
solver.solve(rhs, solution.vector)
solution.x.scatter_forward()

# Another option could be to create a function and set some values to it
rhs = dolfinx.fem.Function(p.V)
with rhs.vector.localForm() as rhs_loc:
    rhs_loc.set(0)
assemble_vector(rhs.vector, some_compiled_form) or skip this
apply_lifting(rhs.vector, [p.a], bcs=[p.get_dirichlet_bcs()])
rhs.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(rhs.vector, p.get_dirichlet_bcs())
solver.solve(rhs.vector, solution.vector)
solution.x.scatter_forward()

"""


class LinearProblem(dolfinx.fem.petsc.LinearProblem):

    """Class for solving a linear variational problem"""

    def __init__(self, domain, V):
        """Initialize domain and FE space

        Parameters
        ----------
        domain : multi.Domain
            The computational domain.
        V : dolfinx.fem.FunctionSpace
            The FE space.

        """
        self.logger = getLogger("multi.problems.LinearProblem")
        self.domain = domain
        self.V = V
        self.u = ufl.TrialFunction(V)
        self.v = ufl.TestFunction(V)
        self._bc_handler = BoundaryConditions(domain.grid, V, domain.facet_markers)

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
    def form_lhs(self):
        """The ufl form of the left hand side"""
        raise NotImplementedError

    @property
    def form_rhs(self):
        """The ufl form of the right hand side"""
        raise NotImplementedError

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

        super().__init__(
            a,
            L,
            bcs,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )

    def assemble_matrix(self, bcs=[]):
        """assemble matrix and apply boundary conditions"""
        self._A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._A, self._a, bcs=bcs)
        self._A.assemble()

    def assemble_vector(self, bcs=[]):
        """assemble vector and apply boundary conditions"""

        with self._b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        dolfinx.fem.petsc.apply_lifting(self._b, [self._a], bcs=[bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self._b, self.bcs)


class LinearElasticityProblem(LinearProblem):
    """class representing a linear elastic problem

    Parameters
    ----------
    domain : multi.domain.Domain
        The computational domain.
    V : dolfin.FunctionSpace
        The finite element space.
    E : float or tuple of float
        Young's modulus of the linear elastic materials.
    NU : float or tuple of float
        Poisson ratio of the linear elastic materials.
    plane_stress : bool, optional
        2d constraint.

    """

    def __init__(
        self, domain, V, E=210e3, NU=0.3, plane_stress=False
    ):
        super().__init__(domain, V)
        assert all(
            [isinstance(E, (float, tuple, list)), isinstance(NU, (float, tuple, list))]
        )
        if isinstance(E, float) and isinstance(NU, float):
            E = (E,)
            NU = (NU,)
            assert domain.cell_markers is None
            self.dx = ufl.dx
        else:
            if len(E) > 1 and domain.cell_markers is None:
                raise KeyError("You need to define mesh tags for multiple materials")
            assert all(
                [
                    len(E) == len(NU),
                    len(E) == np.unique(domain.cell_markers.values).size,
                ]
            )
            # FIXME double check if gmsh cell data starts at 1
            assert np.amin(domain.cell_markers.values) > 0
            mesh = domain.grid
            subdomains = domain.cell_markers
            self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        self.gdim = int(V.element.value_shape)
        assert self.gdim in (1, 2, 3)
        self.materials = [
            LinearElasticMaterial(self.gdim, E=e, NU=nu, plane_stress=plane_stress)
            for e, nu in zip(E, NU)
        ]

    def setup_edge_spaces(self):

        edge_meshes = {}
        try:
            edge_meshes["fine"] = self.domain.fine_edge_grid
        except AttributeError as err:
            raise err("Fine grid partition of the edges does not exist.")

        try:
            edge_meshes["coarse"] = self.domain.coarse_edge_grid
        except AttributeError as err:
            raise err("Coarse grid partition of the edges does not exist.")

        V = self.V
        ufl_element = V.ufl_element()
        family_name = ufl_element.family_name
        degree = ufl_element.degree()

        edge_spaces = {}
        for scale, data in edge_meshes.items():
            edge_spaces[scale] = {}
            if scale == "fine":
                fe = (family_name, degree)
            else:
                fe = (family_name, 1)
            for edge, grid in data.items():
                space = dolfinx.fem.VectorFunctionSpace(grid, fe)
                edge_spaces[scale][edge] = space

        self.edge_spaces = edge_spaces

    def setup_coarse_space(self):
        try:
            coarse_grid = self.domain.coarse_grid
        except AttributeError:
            pass
        V = self.V
        ufl_element = V.ufl_element()
        family_name = ufl_element.family_name
        self.W = dolfinx.fem.VectorFunctionSpace(coarse_grid, (family_name, 1))

    def create_map_from_V_to_L(self):
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

    @property
    def form_lhs(self):
        """get bilinear form a(u, v) of the problem"""
        u = self.u
        v = self.v
        if len(self.materials) > 1:
            return sum(
                [
                    ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx(i + 1)
                    for (i, mat) in enumerate(self.materials)
                ]
            )
        else:
            mat = self.materials[0]
            return ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx

    # FIXME allow for body forces
    @property
    def form_rhs(self):
        """get linear form f(v) of the problem"""
        domain = self.V.mesh
        v = self.v
        zero = dolfinx.fem.Constant(domain, (PETSc.ScalarType(0.0),) * self.gdim)
        rhs = ufl.inner(zero, v) * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs


class TransferProblem(object):
    """General class for transfer problems.

    This class aids the solution of transfer problems given by:

        A(u) = 0 in Ω,
        with homogeneous Dirichlet bcs on Γ_D,
        with arbitrary Dirichlet boundary conditions on Γ_out.

    Additional Neumann bcs can be set via property `neumann`.

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

    Parameters
    ----------
    problem : multi.problems.LinearProblem
        The problem defined on the oversampling domain Ω.
    subdomain_problem : multi.problem.LinearProblem
        The problem defined on the target subdomain.
    gamma_out : callable
        A function that defines facets of Γ_out geometrically.
        `dolfinx.mesh.locate_entities_boundary` is used to determine the facets.
        Γ_out is the part of the boundary of the oversampling domain that does
        not intersect with the boundary of the global domain.
    dirichlet : list of dict or dict, optional
        Homogeneous dirichlet boundary conditions.
        See multi.bcs.BoundaryConditions.add_dirichlet_bc for suitable values.
    source_product : dict, optional
        The inner product to use for the source space. The dictionary should define
        the key `product` and optionally `bcs` and `product_name`.
    range_product : dict, optional
        The inner product to use for the range space. The dictionary should define
        the key `product` and optionally `bcs` and `product_name`.
    remove_kernel : bool, optional
        If True, remove kernel (rigid body modes) from solution.

    """

    @Timer("TransferProblem.__init__")
    def __init__(
        self,
        problem,
        subdomain_problem,
        gamma_out,
        dirichlet=None,
        source_product=None,
        range_product=None,
        remove_kernel=False,
    ):
        self.logger = getLogger("multi.problems.TransferProblem")
        self.problem = problem
        self.subproblem = subdomain_problem
        self.source = FenicsxVectorSpace(problem.V)
        self.range = FenicsxVectorSpace(subdomain_problem.V)
        self.gamma_out = gamma_out
        self.remove_kernel = remove_kernel

        # initialize commonly used quantities
        self._init_bc_gamma_out()
        self._S_to_R = self._make_mapping()

        # initialize fixed set of dirichlet boundary conditions on Γ_D
        if dirichlet is not None:
            if isinstance(dirichlet, (list, tuple)):
                for dirichlet_bc in dirichlet:
                    problem.add_dirichlet_bc(**dirichlet_bc)
            else:
                problem.add_dirichlet_bc(**dirichlet)
            dirichlet = problem.get_dirichlet_bcs()
        self._bc_hom = dirichlet or []

        # ### inner products
        default_product = {"product": None, "bcs": (), "product_name": None}
        source_prod = source_product or default_product
        range_prod = range_product or default_product
        self.source_product = self._get_source_product(**source_prod)
        self.range_product = self._get_range_product(**range_prod)

        # initialize kernel
        if remove_kernel:
            # build null space
            l2_product = self._get_range_product(product="l2")
            self.range_l2_product = l2_product
            self.kernel = build_nullspace(self.range, product=l2_product, gdim=2)

    @Timer("_init_bc_gamma_out")
    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.source.V
        tdim = V.mesh.topology.dim
        fdim = tdim - 1
        dummy = dolfinx.fem.Function(V)

        # determine boundary facets of Γ_out
        facets_Γ_out = dolfinx.mesh.locate_entities_boundary(
            V.mesh, fdim, self.gamma_out
        )
        # determine dofs on Γ_out
        _dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets_Γ_out)
        bc = dolfinx.fem.dirichletbc(dummy, _dofs)

        # only internal use
        self._facets_Γ_out = facets_Γ_out
        self._fdim = fdim
        self._dummy_bc_gamma_out = bc
        self._dofs_Γ_out = _dofs

        # quantities that might be used --> property
        dofs = bc.dof_indices()[0]
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

    @Timer("_make_mapping")
    def _make_mapping(self):
        """builds map from source space to range space"""
        return make_mapping(self.range.V, self.source.V)

    @Timer("discretize_operator")
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

    @Timer("discretize_rhs")
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

        # ddd = dolfinx.fem.locate_dofs_topological(tp.source.V, tp.fdim, tp.facets_Γ_out)
        # bc_inhom = dolfinx.fem.dirichletbc(array, ddd, tp.source.V) # most likely
        # because random values are created using numpy

        dofs = self.bc_dofs_gamma_out
        _dofs = self._dofs_Γ_out
        p = self.problem
        f = dolfinx.fem.Function(p.V)
        f.x.array[dofs] = boundary_values
        bc_inhom = dolfinx.fem.dirichletbc(f, _dofs)

        p.assemble_vector(bcs=[bc_inhom])

    # @Timer("generate_boundary_data")
    # def generate_boundary_data(self, values):
    #     """generate boundary data g in V(Γ_out)"""
    #     bc_dofs = self.bc_dofs_gamma_out
    #     assert values.shape[1] == len(bc_dofs)
    #     D = np.zeros((len(values), self.source.dim))
    #     D[:, bc_dofs] = values
    #     return self.source.from_numpy(D)

    @Timer("generate_random_boundary_data")
    def generate_random_boundary_data(
        self, count, distribution="normal", random_state=None, seed=None, **kwargs
    ):
        """generate random values shape (count, num_dofs_Γ_out)"""

        bc_dofs = self.bc_dofs_gamma_out  # actual size of the source space
        assert random_state is None or seed is None
        random_state = get_random_state(random_state, seed)
        values = _create_random_values(
            (count, bc_dofs.size), distribution, random_state, **kwargs
        )

        return values

    @Timer("TransferProblem.solve")
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
        u = dolfinx.fem.Function(p.V)  # full space
        u_in = dolfinx.fem.Function(self.range.V)  # target subdomain
        U = self.range.empty()  # VectorArray to store u_in

        # construct rhs from boundary data
        for array in boundary_values:
            self.discretize_rhs(array)
            solver.solve(rhs, u.vector)
            u.x.scatter_forward()

            # restrict full solution to target subdomain
            u_in.interpolate(u)
            U.append(self.range.make_array([u_in.vector.copy()]))

        if self.remove_kernel:
            return orthogonal_part(self.kernel, U, self.range_l2_product, orth=True)
        else:
            return U

    @Timer("_get_source_product")
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
            # FIXME figure out how to do this with PETSc.Mat and
            # use FenicsxMatrixOperator instead of NumpyMatrixOperator
            full_matrix = csc_matrix(M.getValuesCSR()[::-1], shape=M.size)
            dofs = self.bc_dofs_gamma_out
            source_matrix = full_matrix[dofs, :][:, dofs]
            source_product = NumpyMatrixOperator(source_matrix, name=product_name)
            return source_product
        else:
            return None

    @Timer("_get_range_product")
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


class MultiscaleProblem(object):
    """base class to handle definition of a multiscale problem"""

    def __init__(self, coarse_grid_path, fine_grid_path):
        self.coarse_grid_path = pathlib.Path(coarse_grid_path)
        self.fine_grid_path = pathlib.Path(fine_grid_path)
        self._setup_grids()

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

    def _setup_grids(self):
        """create coarse and fine grid"""
        domain, ct, ft = gmshio.read_from_msh(
            self.coarse_grid_path.as_posix(), MPI.COMM_WORLD, gdim=2
        )
        self.coarse_grid = StructuredQuadGrid(domain, ct, ft)

        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, self.fine_grid_path.as_posix(), "r"
        ) as xdmf:
            fine_domain = xdmf.read_mesh(name="Grid")
            fine_ct = xdmf.read_meshtags(fine_domain, name="Grid")

        boundaries = self.boundaries
        if boundaries is not None:
            from multi.preprocessing import create_facet_tags

            fine_ft, marked_boundaries = create_facet_tags(fine_domain, boundaries)
        else:
            fine_ft = None
        self.fine_grid = Domain(
            fine_domain, cell_markers=fine_ct, facet_markers=fine_ft
        )

    def setup_fe_spaces(self, family="P"):
        """create FE spaces on coarse and fine grid"""
        try:
            degree = self.degree
        except AttributeError as err:
            raise err("You need to set the degree of the problem first")
        self.W = dolfinx.fem.VectorFunctionSpace(self.coarse_grid.grid, (family, 1))
        self.V = dolfinx.fem.VectorFunctionSpace(self.fine_grid.grid, (family, degree))

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
