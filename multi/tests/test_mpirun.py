import dolfinx
from pymor.core.defaults import set_defaults, print_defaults
from pymor.tools import mpi


if __name__ == "__main__":
    set_defaults({"pymor.tools.mpi.event_loop_settings.auto_launch": False})
    print_defaults()
    if mpi.parallel:
        print("hello from the other side")
        domain = dolfinx.mesh.create_unit_interval(mpi.comm, 10)
    else:
        print("hello")


"""01.10.2022

• setting the auto launch to False did not help
• was able to run demo fenics_nonlinear.py (old dolfin) in parallel with
    mpirun -n 4 python fenics_nonlinear.py
  But this gave me Warnings: no restricted operator available (which is not the case in serial).

"""
