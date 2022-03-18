import numpy as np
import dolfin as df
from multi.misc import make_mapping
from pymor.core.logger import getLogger
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.bindings.fenics import FenicsMatrixOperator
from scipy.sparse import csc_matrix


class OversamplingProblem(object):

    """General class for oversampling problems.

    This class aids the solution of oversampling problems given by:

        A(u) = 0 in Ω,
        with homogeneous dirichlet bcs on Γ_D,
        with homogeneous neumann bcs on Γ_N,
        with inhomogeneous neumann bcs on Γ_N_inhom,
        with arbitrary dirichlet boundary conditions on Γ_out.

    Here, we are only interested in the solution u restricted to the
    space defined on a target subdomain Ω_in ⊂ Ω.

    The boundaries Γ_D, Γ_N and Γ_N_inhom are the part of ∂Ω that
    intersects with the (respective dirichlet or neumann) boundary
    of the global domain Ω_gl ( Ω ⊂ Ω_gl).
    The boundary Γ_out is the part of ∂Ω that does not intersect with ∂Ω_gl.

    Note: The above problem can be formulated as a transfer operator which
    maps the boundary data on Γ_out (source space) to the solution u in
    the (range) space defined on Ω_in. Since fenics (version 2019.1.0) does
    not allow to define function spaces on some part of the boundary of a domain,
    the full space is defined as the source space. The range space is the
    space defined on Ω_in.

    Parameters
    ----------
    problem : multi.linear_elasticity.LinearElasticityProblem
        The problem defined on the oversampling domain Ω.
    subdomain_problem : multi.linear_elasticity.LinearElasticityProblem
        The problem defined on the target subdomain Ω_in.
    gamma_out : df.SubDomain
        The part of the boundary of the oversampling domain that does
        not intersect with the boundary of the global domain.
    dirichlet : list of dict or dict, optional
        Homogeneous or inhomogeneous dirichlet boundary conditions.
        See multi.bcs.MechanicsBCs.add_bc for suitable values.
    neumann : list of dict or dict, optional
        Inhomogeneous neumann boundary conditions.
        See multi.bcs.MechanicsBCs.add_force for suitable values.
    solver_options : dict, optional
        The user is required to use pymor style options ({'inverse': options}),
        because the value of solver_options is directly passed to FenicsMatrixOperator.
        See https://github.com/pymor/pymor/blob/main/src/pymor/operators/interface.py#L32-#L41


    """

    def __init__(
        self,
        problem,
        subdomain_problem,
        gamma_out,
        dirichlet=None,
        neumann=None,
        solver_options=None,
    ):
        self._logger = getLogger("multi.oversampling.OversamplingProblem")
        self.problem = problem
        self.subdomain_problem = subdomain_problem
        self.source = problem.source
        self.range = subdomain_problem.range
        self.gamma_out = gamma_out
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.solver_options = solver_options
        # initialize commonly used quantities
        self._init_bc_gamma_out()
        self._S_to_R = self._make_mapping()

    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.problem.V
        dummy = df.Function(V)
        self._bc_gamma_out = df.DirichletBC(V, dummy, self.gamma_out)
        self._bc_dofs_gamma_out = list(self._bc_gamma_out.get_boundary_values().keys())

    def _make_mapping(self):
        """builds map from source space to range space"""
        return make_mapping(self.subdomain_problem.V, self.problem.V)

    def _discretize_operator(self):
        """discretize the operator"""
        self._logger.info(
            f"Discretizing left hand side of the problem (size={self.problem.V.dim()})."
        )
        matrix = df.PETScMatrix()
        df.assemble(self.problem.get_lhs(), tensor=matrix)
        # make copy and wrap as FenicsMatrixOperator for construction of rhs
        V = self.problem.V
        # A refers to full operator without bcs applied
        self._A = FenicsMatrixOperator(
            matrix.copy(), V, V, solver_options=self.solver_options, name="A"
        )

        # ### apply bcs to operator
        # make sure there are no unwanted bcs present
        self.problem.bc_handler.remove_bcs()
        if self.dirichlet is not None:
            if isinstance(self.dirichlet, list):
                for dirichlet_bc in self.dirichlet:
                    self.problem.bc_handler.add_bc(**dirichlet_bc)
            else:
                self.problem.bc_handler.add_bc(**self.dirichlet)
        bcs = [self._bc_gamma_out] + self.problem.bc_handler.bcs()
        dummy = df.Function(V)
        for bc in bcs:
            bc.zero_columns(matrix, dummy.vector(), 1.0)

        # A_0 refers to operator with bcs applied
        self.operator = FenicsMatrixOperator(
            matrix, V, V, solver_options=self.solver_options, name="A_0"
        )

    def _discretize_neumann(self):
        """discretize inhomogeneous neumann bc(s)"""
        self.problem.bc_handler.remove_forces()
        if self.neumann is not None:
            if isinstance(self.neumann, list):
                for force in self.neumann:
                    self.problem.bc_handler.add_force(**force)
            else:
                self.problem.bc_handler.add_force(**self.neumann)

        # will always be null vector in case neumann is None see LinearElasticityProblem.get_rhs
        rhs = self.source.make_array([df.assemble(self.problem.get_rhs())])
        self._f_ext = rhs

    def _discretize_rhs(self, boundary_data):
        """discretize the right hand side"""
        R = boundary_data
        assert R in self.source

        if not hasattr(self, "_f_ext"):
            # assemble external force only once
            self._discretize_neumann()
        rhs = self._f_ext

        # subtract g(x_i) times the i-th column of A from the rhs
        # rhs = rhs - self._A.apply(R)
        AR = self._A.apply(R)
        AR.axpy(-1, rhs)
        rhs = -AR

        # set g(x_i) for i-th dof in rhs
        bc_dofs = list(self._bc_gamma_out.get_boundary_values().keys())
        bc_vals = R.dofs(bc_dofs)
        # workaround
        rhs_array = rhs.to_numpy()
        rhs_array[:, bc_dofs] = bc_vals

        return self.source.from_numpy(rhs_array)

    def generate_random_boundary_data(
        self, count, distribution="normal", random_state=None, seed=None
    ):
        """generate random boundary data"""
        # initialize
        D = np.zeros((count, self.source.dim))

        # use NumpyVectorSpace to create random values
        bc_dofs = self._bc_dofs_gamma_out
        space = NumpyVectorSpace(len(bc_dofs))
        random_values = space.random(
            count, distribution=distribution, random_state=random_state, seed=seed
        )
        # set random data at boundary dofs
        D[:, bc_dofs] = random_values.to_numpy()
        return self.source.from_numpy(D)

    def solve(self, boundary_data):
        """solve the problem for boundary_data

        Parameters
        ----------
        boundary_data : VectorArray
            Vectors in FenicsVectorSpace(problem.V) with DoF entries holding
            values of boundary data on Γ_out and zero elsewhere.

        Returns
        -------
        U_in : VectorArray
            The solutions in the range space.
        """
        if not hasattr(self, "operator"):
            self._discretize_operator()
        # construct rhs from boundary data
        rhs = self._discretize_rhs(boundary_data)
        self._logger.info(f"Solving OversamplingProblem for {len(rhs)} vectors.")
        U = self.operator.apply_inverse(rhs)
        return self.range.from_numpy(U.dofs(self._S_to_R))

    def get_source_product(self, product=None):
        """get source product

        Parameters
        ----------
        product : str, optional
            The inner product to use.

        Returns
        -------
        source_product : NumpyMatrixOperator or None
        """
        if product is not None:
            # compute source product
            matrix = self.problem.get_product(name=product, bcs=False)
            M = df.as_backend_type(matrix).mat()
            # FIXME figure out how to use (take slice of) dolfin matrix directly?
            full_matrix = csc_matrix(M.getValuesCSR()[::-1], shape=M.size)
            dofs = self._bc_dofs_gamma_out
            source_matrix = full_matrix[dofs, :][:, dofs]
            source_product = NumpyMatrixOperator(
                source_matrix, name=f"{product}_product"
            )
            return source_product
        else:
            return None

    def get_range_product(self, product=None):
        """get range product

        Parameters
        ----------
        product : str, optional
            The inner product to use.

        Returns
        -------
        range_product : FenicsMatrixOperator or None
        """
        if product is not None:
            # compute range product
            matrix = self.subdomain_problem.get_product(name=product, bcs=False)
            V = self.range.V
            range_product = FenicsMatrixOperator(
                matrix, V, V, name=f"{product}_product"
            )
            return range_product
        else:
            return None
