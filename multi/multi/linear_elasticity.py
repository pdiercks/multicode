import logging
import dolfin as df
import numpy as np
from multi.bcs import BoundaryConditions
from multi.product import InnerProduct
from pymor.bindings.fenics import FenicsVectorSpace


class LinearElasticMaterial:
    """class representing linear elastic isotropic materials

    Parameters
    ----------
    dim
        Spatial dimension of the problem.
    E=210e3
        Young's modulus.
    NU=0.3
        Poisson ratio.
    plane_stress=False
        Constraints to use in case dim=2.

    Attributes
    ----------
    dim              Spatial dimension of the problem.
    plane_stress     Constraints to use in case dim=2.
    E                Young's modulus.
    NU               Poisson ratio.
    lambda_1         First lame constant.
    lambda_2         Second lame constant.
    """

    def __init__(self, dim, E=210e3, NU=0.3, plane_stress=False):
        self.dim = dim
        self.plane_stress = plane_stress
        self.E = E
        self.NU = NU
        self.lambda_1 = E * NU / ((1 + NU) * (1 - 2 * NU))
        self.lambda_2 = E / 2 / (1 + NU)

    def get_matrix(self, u, v, dx):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        epsu = self.eps(u)
        epsv = self.eps(v)
        return df.assemble(
            (2 * lambda_2 * df.inner(epsu, epsv) + lambda_1 * df.tr(epsu) * df.tr(epsv))
            * dx
        )

    def sigma(self, displacement):
        eps = self.eps(displacement)
        return self.lambda_1 * df.tr(eps) * df.Identity(3) + 2 * self.lambda_2 * eps

    def eps(self, displacement):
        d = self.dim
        e = df.sym(df.grad(displacement))
        if d == 1:
            return df.as_tensor([[e[0, 0], 0, 0], [0, 0, 0], [0, 0, 0]])
        elif d == 2 and not self.plane_stress:
            return df.as_tensor(
                [[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]]
            )
        elif d == 2 and self.plane_stress:
            ezz = (
                -self.lambda_1
                / (2.0 * self.lambda_2 + self.lambda_1)
                * (e[0, 0] + e[1, 1])
            )
            return df.as_tensor(
                [[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, ezz]]
            )
        elif d == 3:
            return e
        else:
            AttributeError("Spatial Dimension is not set.")


class LinearElasticityProblem:
    """a linear elasticity problem

    Parameters
    ----------
    domain
        The computational domain.
    V
        The finite element space.
    E : float or tuple of float
        Young's modulus of the linear elastic materials.
    NU : float or tuple of float
        Poisson ratio of the linear elastic materials.
    plane_stress : bool, optional
        2d constraint.

    """

    def __init__(self, domain, V, E=210e3, NU=0.3, plane_stress=False):
        assert all(
            [isinstance(E, (float, tuple, list)), isinstance(NU, (float, tuple, list))]
        )
        if isinstance(E, float) and isinstance(NU, float):
            E = (E,)
            NU = (NU,)
            assert not domain.subdomains
            self.dx = df.dx
        else:
            if not domain.subdomains and len(E) > 1:
                raise KeyError(
                    "You need to define a df.MeshFunction for multiple materials"
                )
            assert all(
                [len(E) == len(NU), len(E) == np.unique(domain.subdomains.array()).size]
            )
            # pygmsh version 6.1.1 convention
            assert np.amin(domain.subdomains.array()) > 0
            mesh = domain.mesh
            subdomains = domain.subdomains
            self.dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)
        self.logger = logging.getLogger("LinearElasticityProblem")
        self.domain = domain
        self.V = V
        self.source = FenicsVectorSpace(V)
        self.range = FenicsVectorSpace(V)
        self.gdim = V.element().geometric_dimension()
        self.u = df.TrialFunction(V)
        self.v = df.TestFunction(V)
        self.bc_handler = MechanicsBCs(domain, V)
        self.materials = [
            LinearElasticMaterial(self.gdim, E=e, NU=nu, plane_stress=plane_stress)
            for e, nu in zip(E, NU)
        ]

    def get_lhs(self):
        """get bilinear form a(u, v) of the problem"""
        u = self.u
        v = self.v
        if len(self.materials) > 1:
            return sum(
                [
                    df.inner(mat.sigma(u), mat.eps(v)) * self.dx(i + 1)
                    for (i, mat) in enumerate(self.materials)
                ]
            )
        else:
            mat = self.materials[0]
            return df.inner(mat.sigma(u), mat.eps(v)) * self.dx

    def get_rhs(self, body_forces=None):
        """get linear form f(v) of the problem"""
        v = self.v
        zero = (0.0,) * self.gdim
        rhs = df.dot(df.Constant(zero), v) * df.dx
        if body_forces is not None:
            if len(self.materials) > 1:
                assert isinstance(body_forces, (list, tuple))
                assert len(body_forces) == len(self.materials)
                for i in range(len(self.materials)):
                    rhs += df.dot(body_forces[i], v) * self.dx(i + 1)
            else:
                rhs += df.dot(body_forces, v) * self.dx

        if self.bc_handler.has_forces():
            rhs += self.bc_handler.boundary_forces()

        return rhs

    def get_product(self, name="energy", bcs=True):
        """get inner product associated with V of the problem

        Parameters
        ----------
        name : str, optional
            The name of the inner product.
        bcs : bool, optional
            If True apply BCs of problem to the inner product.

        Returns
        -------
        dolfin.Matrix

        """
        if bcs:
            bcs = self.bc_handler.bcs()
        else:
            bcs = ()
        if name == "energy":
            product = InnerProduct(self.V, name, bcs=bcs, form=self.get_lhs())
        else:
            product = InnerProduct(self.V, name, bcs=bcs)
        return product.assemble()

    def solve(
        self,
        u=None,
        solver_parameters={"linear_solver": "default", "preconditioner": "default"},
    ):
        """solve a(u, v) = f(v)"""
        a = self.get_lhs()
        L = self.get_rhs()
        bcs = self.bc_handler.bcs()
        if len(bcs) < 1:
            self.logger.warning("No dirichlet bcs defined for this problem...")
        if u:
            df.solve(a == L, u, bcs, solver_parameters=solver_parameters)
            return
        else:
            solution = df.Function(self.V)
            df.solve(a == L, solution, bcs, solver_parameters=solver_parameters)
            return solution
