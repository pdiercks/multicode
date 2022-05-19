"""
periodic homogenization of a given RCE type

The example is taken from
https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html
written by Jeremy Bleyer.

Usage:
    homogenization.py [options] RCE DEG MAT

Arguments:
    RCE       The RCE mesh (incl. extension).
    DEG       The degree of the FE space.
    MAT       Material metadata (.yml).

Options:
    -h, --help               Show this message.
    --plot-mesh              Plot the mesh.
    --plot-exy               Plot solution for the case Exy.
"""

import sys
from docopt import docopt
from pathlib import Path
import yaml
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RCE"] = Path(args["RCE"])
    args["DEG"] = int(args["DEG"])
    args["MAT"] = Path(args["MAT"])
    return args


def main(args):
    args = parse_arguments(args)
    homogenize(args)


def homogenize(args):
    """compute homogenized youngs modulus and poisson ratio and update material.yml"""
    mesh = df.Mesh()
    mvc = df.MeshValueCollection("size_t", mesh)

    with df.XDMFFile(args["RCE"].as_posix()) as f:
        f.read(mesh)
        f.read(mvc, "gmsh:physical")

    subdomains = df.MeshFunction("size_t", mesh, mvc)
    if args["--plot-mesh"]:
        plt.figure(1)
        df.plot(subdomains)
        plt.show()

    # unit cell width
    a = abs(np.amax(mesh.coordinates()[:, 0]) - np.amin(mesh.coordinates()[:, 0]))
    # unit cell height
    b = abs(np.amax(mesh.coordinates()[:, 1]) - np.amin(mesh.coordinates()[:, 1]))
    c = 0.0  # horizontal offset of top boundary
    vol = a * b  # unit cell volume
    # we define the unit cell vertices coordinates for later use
    vertices = np.array([[0.0, 0.0], [a, 0.0], [a + c, b], [c, b]])

    # class used to define the periodic boundary map
    class PeriodicBoundary(df.SubDomain):
        def __init__(self, vertices, tolerance=df.DOLFIN_EPS):
            """vertices stores the coordinates of the 4 unit cell corners"""
            df.SubDomain.__init__(self, tolerance)
            self.tol = tolerance
            self.vv = vertices
            self.a1 = (
                self.vv[1, :] - self.vv[0, :]
            )  # first vector generating periodicity
            self.a2 = (
                self.vv[3, :] - self.vv[0, :]
            )  # second vector generating periodicity
            # check if UC vertices form indeed a parallelogram
            assert np.linalg.norm(self.vv[2, :] - self.vv[3, :] - self.a1) <= self.tol
            assert np.linalg.norm(self.vv[2, :] - self.vv[1, :] - self.a2) <= self.tol

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the
            # bottom-right or top-left vertices
            return bool(
                (
                    df.near(
                        x[0],
                        self.vv[0, 0] + x[1] * self.a2[0] / self.vv[3, 1],
                        self.tol,
                    )
                    or df.near(
                        x[1],
                        self.vv[0, 1] + x[0] * self.a1[1] / self.vv[1, 0],
                        self.tol,
                    )
                )
                and (
                    not (
                        (
                            df.near(x[0], self.vv[1, 0], self.tol)
                            and df.near(x[1], self.vv[1, 1], self.tol)
                        )
                        or (
                            df.near(x[0], self.vv[3, 0], self.tol)
                            and df.near(x[1], self.vv[3, 1], self.tol)
                        )
                    )
                )
                and on_boundary
            )

        def map(self, x, y):
            if df.near(x[0], self.vv[2, 0], self.tol) and df.near(
                x[1], self.vv[2, 1], self.tol
            ):  # if on top-right corner
                y[0] = x[0] - (self.a1[0] + self.a2[0])
                y[1] = x[1] - (self.a1[1] + self.a2[1])
            elif df.near(
                x[0], self.vv[1, 0] + x[1] * self.a2[0] / self.vv[2, 1], self.tol
            ):  # if on right boundary
                y[0] = x[0] - self.a1[0]
                y[1] = x[1] - self.a1[1]
            else:  # should be on top boundary
                y[0] = x[0] - self.a2[0]
                y[1] = x[1] - self.a2[1]

    with open(args["MAT"], "r") as instream:
        material = yaml.safe_load(instream)

    Emm, Eii = material["Material parameters"]["E"]["value"]
    NUmm, NUii = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]
    material_parameters = [(Emm, NUmm), (Eii, NUii)]
    nphases = len(material_parameters)

    def eps(v):
        return df.sym(df.grad(v))

    def sigma(v, i, Eps):
        E, nu = material_parameters[i]
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        if plane_stress:
            lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)
        return lmbda * df.tr(eps(v) + Eps) * df.Identity(2) + 2 * mu * (eps(v) + Eps)

    Ve = df.VectorElement("CG", mesh.ufl_cell(), args["DEG"])
    Re = df.VectorElement("R", mesh.ufl_cell(), 0)
    W = df.FunctionSpace(
        mesh,
        df.MixedElement([Ve, Re]),
        constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10),
    )

    v_, lamb_ = df.TestFunctions(W)
    dv, dlamb = df.TrialFunctions(W)
    w = df.Function(W)
    dx = df.Measure("dx")(subdomain_data=subdomains)
    # dx(1) - matrix, dx(2) - inclusion

    Eps = df.Constant(((0, 0), (0, 0)))
    F = sum([df.inner(sigma(dv, i, Eps), eps(v_)) * dx(i + 1) for i in range(nphases)])
    a, L = df.lhs(F), df.rhs(F)
    a += df.dot(lamb_, dv) * dx + df.dot(dlamb, v_) * dx

    def macro_strain(i):
        """returns the macroscopic strain for the 3 elementary load cases"""
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[i] = 1
        return np.array(
            [[Eps_Voigt[0], Eps_Voigt[2] / 2.0], [Eps_Voigt[2] / 2.0, Eps_Voigt[1]]]
        )

    def stress2Voigt(s):
        return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])

    Chom = np.zeros((3, 3))
    for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
        print("Solving {} case...".format(case))
        Eps.assign(df.Constant(macro_strain(j)))
        df.solve(a == L, w, [], solver_parameters={"linear_solver": "mumps"})
        (v, lamb) = df.split(w)
        Sigma = np.zeros((3,))
        for k in range(3):
            Sigma[k] = (
                df.assemble(
                    sum(
                        [
                            stress2Voigt(sigma(v, i, Eps))[k] * dx(i + 1)
                            for i in range(nphases)
                        ]
                    )
                )
                / vol
            )
        Chom[j, :] = Sigma

    print("Homogenized Stiffness:\n")
    print(np.array_str(Chom, precision=2))

    lmbda_hom = Chom[0, 1]
    mu_hom = Chom[2, 2]
    iso_err = np.abs(Chom[0, 0] - lmbda_hom - 2 * mu_hom) / Chom[0, 0]

    csymm = (Chom + Chom.T) / 2
    cskew = (Chom - Chom.T) / 2
    sym_mea = (np.linalg.norm(csymm) - np.linalg.norm(cskew)) / (
        np.linalg.norm(csymm) + np.linalg.norm(cskew)
    )
    summary = f"""Summary for Stiffness:
        relative error isotropy:        {iso_err}
        measure for symmetry:           {sym_mea}"""
    print(summary)

    # if plane_stress:
    #     lmbda_hom = - 2 * mu_hom * lmbda_hom / (lmbda_hom - 2 * mu_hom)
    # E_hom = mu_hom * (3 * lmbda_hom + 2 * mu_hom) / (lmbda_hom + mu_hom)
    # nu_hom = lmbda_hom / (lmbda_hom + mu_hom) / 2
    # print("Apparent Young modulus:", E_hom)
    # print("Apparent Poisson ratio:", nu_hom)

    rce_type = args["RCE"].parent.stem
    # get value of key; return empty dict if None
    homogenized_stiffness = material.get("Homogenized stiffness", {})
    homogenized_stiffness.update({rce_type: Chom.tolist()})
    material.update({"Homogenized stiffness": homogenized_stiffness})
    with open(args["MAT"], "w") as outStream:
        yaml.safe_dump(material, outStream)

    # plotting deformed unit cell with total displacement u = Eps*y + v
    if args["--plot-exy"]:
        y = df.SpatialCoordinate(mesh)
        plt.figure(2)
        p = df.plot(0.5 * (df.dot(Eps, y) + v), mode="displacement", title=case)
        plt.colorbar(p)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
