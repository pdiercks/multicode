# This code was written by Thomas Titscher.
# By the courtesy of Thomas Titscher it was slightly modified and used.
import numpy as np
import edmd


class FullerCurve(edmd.GradingCurve):
    def __init__(self, d_min=2, d_max=16, q=0.5, N=3):
        """
        Fuller-Thompson curve with the CDF
          f(d) = (d / d_max)**q

        evaluated in the range
          d_min to d_max
        with
          N
        classes
        """
        super().__init__()

        self.d_max = d_max
        self.q = q
        self.ds = np.geomspace(d_min, d_max, N + 1)

        for i in range(N):
            d_min = self.ds[i]
            d_max = self.ds[i + 1]
            fraction = self.fuller(d_max) - self.fuller(d_min)
            self.add_grading_class(d_min, d_max, fraction)

    def fuller(self, d):
        """evaluates f(d) = (d / d_max)**q"""
        return (d / self.d_max) ** self.q

    def show(self, volume, seed=0):
        import matplotlib.pyplot as plt

        radii = self.sample(volume, seed)
        plt.plot(self.ds, [self.fuller(d) for d in self.ds], "-kx", label="reference")

        ds, fs, V = [], [], 0
        for r in reversed(radii):
            ds.append(2 * r)
            V += 4.0 / 3.0 * np.pi * r ** 3
            fs.append(V)

        V_offset = volume - V

        plt.plot(ds, (np.asarray(fs) + V_offset) / volume, label="sampled")
        plt.xscale("log")
        plt.legend()
        plt.show()


class Stats:
    def __init__(self, sim):
        self.sim = sim

    def header(self):
        return "  time  |  events |   vol%  |  rate "

    def __call__(self):
        s = "{:7.4f} | ".format(self.sim.t())
        s += self.human_format(self.sim.stats.n_events) + " | "
        s += "{:6.3f}%".format(self.sim.stats.pf * 100) + " | "
        s += self.human_format(self.sim.stats.collisionrate) + "/s"
        return s

    def human_format(self, number):
        from math import log, floor

        """
        thanks to https://stackoverflow.com/a/45478574
        """
        units = [" ", "K", "M", "G", "T", "P"]

        if number == 0:
            magnitude = 0
        else:
            magnitude = int(floor(log(number, 1000)))

        return "{:6.2f}{}".format(number / 1000 ** magnitude, units[magnitude])


class VerboseSimulation(edmd.Simulation):
    def __init__(self, spheres, box, seed):
        super().__init__(spheres, box, seed)
        self.stop = []

    def add_stop_time(self, value):
        def f():
            return self.t() < value

        self.stop.append(f)

    def add_stop_pf(self, value):
        def f():
            return self.stats.pf < value

        self.stop.append(f)

    def add_stop_rate(self, value):
        def f():
            return self.stats.collisionrate < value

        self.stop.append(f)

    def _do_continue(self):
        cont = True
        for f in self.stop:
            cont = cont and f()
        return cont

    def run(self, step_length=10, t_end=0.3):
        info = Stats(self)
        print(info.header())

        N = len(self.radii)
        self.process(0)
        print(info())

        i = 0
        while self._do_continue():
            self.process(N * step_length)
            i += 1
            if i % 10 == 0:
                self.synchronize(True)
            print(info())


def maximize_particle_distance(grading_curve, box, phi, T=0.1, seed=6174):
    v_sphere = box.volume() * phi

    spheres = []
    v = edmd.MaxwellBoltzmann(seed)

    radii = grading_curve.sample(v_sphere, seed)
    for r in radii:
        s = edmd.Sphere()
        s.r = r
        s.gr = 1
        s.m = 1
        s.v = v.vector(T)
        spheres.append(s)

    N = len(spheres)
    sim = VerboseSimulation(spheres, box, seed)

    # sim.add_stop_time(0.9)
    sim.add_stop_rate(1e9)
    sim.add_stop_pf(0.7)

    sim.run(step_length=200)
    sim.update_spheres()

    return (sim.positions, radii)


def create_gmsh():
    f = FullerCurve(d_min=3, d_max=4, q=0.7)
    a = 20
    box = edmd.Cube(a, a, a)
    s = maximize_particle_distance(f, box, 0.6, T=42)

    opts = edmd.GmshOptions()
    opts.slice = a / 2
    opts.interface_thickness = 0.0
    opts.gmsh_order = 1
    opts.recombine = False

    # mesh size parameters
    opts.ntransfinite = 0
    opts.mesh_size_matrix = 1.0
    opts.mesh_size_aggregates = 1.0

    g = edmd.GmshWriter(s, box, opts)
    g.write_msh("rce.msh", "rce.geo")

    import meshio

    geometry = meshio.read("rce.msh")
    # prune z:
    if opts.slice != 0:
        geometry.points = geometry.points[:, :2]

    meshio.write(
        "rce.xdmf",
        meshio.Mesh(
            points=geometry.points, cells=geometry.cells, cell_data=geometry.cell_data
        ),
    )


if __name__ == "__main__":
    create_gmsh()
