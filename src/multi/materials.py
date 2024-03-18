from typing import Union

import ufl
from dolfinx.fem import Constant


class LinearElasticMaterial:
    """Class representing linear elastic isotropic materials."""

    def __init__(self, gdim: int, E: Union[float, Constant], NU: Union[float, Constant], plane_stress: bool = False):
        """Initializes the material.

        Args:
            gdim: Geometrical dimension of the problem.
            E: Young's modulus.
            NU: Poisson ratio.
            plane_stress: If True, do plane stress.

        """
        self.gdim = gdim
        self.E = E
        self.NU = NU
        self.plane_stress = plane_stress
        self.lambda_1 = E * NU / (1 + NU) / (1 - 2 * NU)
        self.lambda_2 = E / 2 / (1 + NU)

    def sigma(self, displacement: ufl.Argument):
        eps = self.eps(displacement)
        return self.lambda_1 * ufl.tr(eps) * ufl.Identity(3) + 2 * self.lambda_2 * eps

    def eps(self, displacement: ufl.Argument):
        d = self.gdim
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
        else:
            return e
