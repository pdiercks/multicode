import ufl


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
            (
                2 * lambda_2 * ufl.inner(epsu, epsv)
                + lambda_1 * ufl.tr(epsu) * ufl.tr(epsv)
            )
            * dx
        )

    def sigma(self, displacement):
        eps = self.eps(displacement)
        return self.lambda_1 * ufl.tr(eps) * ufl.Identity(3) + 2 * self.lambda_2 * eps

    def eps(self, displacement):
        d = self.dim
        e = ufl.sym(ufl.grad(displacement))
        if d == 1:
            return ufl.as_tensor([[e[0, 0], 0, 0], [0, 0, 0], [0, 0, 0]])
        elif d == 2 and not self.plane_stress:
            return ufl.as_tensor(
                [[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]]
            )
        elif d == 2 and self.plane_stress:
            ezz = (
                -self.lambda_1
                / (2.0 * self.lambda_2 + self.lambda_1)
                * (e[0, 0] + e[1, 1])
            )
            return ufl.as_tensor(
                [[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, ezz]]
            )
        elif d == 3:
            return e
        else:
            AttributeError("Spatial Dimension is not set.")
