"""
fom_dolfinx.py — full-order model of the cylinder-wake benchmark (DFG 2D-2, Re = 100).

Incompressible Navier–Stokes in the channel [0, 2.2] x [0, 0.41] around a cylinder of radius 0.05 centred at
(0.2, 0.2) (Schäfer & Turek 1996, test case 2D-2): density 1, viscosity 1e-3, parabolic inflow of peak 1.5 (mean
velocity 1, so Re = 1 * 0.1 / 1e-3 = 100), no slip on the walls and the cylinder, zero pressure at the outflow.
Taylor–Hood P2–P1 elements; incremental pressure-correction scheme (IPCS) with Crank–Nicolson diffusion and
Adams–Bashforth convection. The flow starts from rest; the inflow is ramped smoothly over the first 0.5 time units.
The slight asymmetry of the geometry (cylinder centre 0.2, channel centre 0.205) triggers vortex shedding.

The saved full-order state is the velocity (u, v) sampled on a uniform 0.01 grid over the whole channel
(221 x 42 points, zero inside the cylinder), i.e. what a PIV measurement would record: 2 x 9282 = 18,564 values.
Drag and lift coefficients are saved at every step for validation against the benchmark reference
(Cd_max 3.2200-3.2400, Cl_max 0.9900-1.0100, St 0.2950-0.3050).

Run in the FEniCSx environment (serial):
    python fom_dolfinx.py --out cylinder_fom.npz --T 20 --dt 6.25e-4 --save_every 8
--nu sets the viscosity (default 1e-3, Re 100); --nu 2.5e-4 gives Re 400 (higher-mode variant, DESIGN.md).
"""
import argparse
import time

import gmsh
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, geometry
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, create_matrix, create_vector,
                               set_bc)
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc

L, H, CX, CY, R = 2.2, 0.41, 0.2, 0.2, 0.05
NU, U_MAX, U_MEAN, D = 1e-3, 1.5, 1.0, 0.1
INLET, OUTLET, WALLS, CYLINDER = 2, 3, 4, 5
GRID_STEP = 0.01


def build_mesh(h_cyl, h_wake, h_far):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, L, H)
    disk = occ.addDisk(CX, CY, 0, R, R)
    occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
    groups = {INLET: [], OUTLET: [], WALLS: [], CYLINDER: []}
    for dim, tag in gmsh.model.getBoundary(surfaces, oriented=False):
        x, y, _ = occ.getCenterOfMass(dim, tag)
        if np.isclose(x, 0):
            groups[INLET].append(tag)
        elif np.isclose(x, L):
            groups[OUTLET].append(tag)
        elif np.isclose(y, 0) or np.isclose(y, H):
            groups[WALLS].append(tag)
        else:
            groups[CYLINDER].append(tag)
    for marker, tags in groups.items():
        gmsh.model.addPhysicalGroup(1, tags, marker)
    field = gmsh.model.mesh.field
    dist = field.add("Distance")
    field.setNumbers(dist, "CurvesList", groups[CYLINDER])
    field.setNumber(dist, "Sampling", 200)
    near = field.add("Threshold")
    field.setNumber(near, "InField", dist)
    field.setNumber(near, "SizeMin", h_cyl)
    field.setNumber(near, "SizeMax", h_wake)
    field.setNumber(near, "DistMin", 0.25 * R)
    field.setNumber(near, "DistMax", 2 * R)
    wake = field.add("Box")
    field.setNumber(wake, "VIn", h_wake)
    field.setNumber(wake, "VOut", h_far)
    field.setNumber(wake, "XMin", CX - 2 * R)
    field.setNumber(wake, "XMax", L)
    field.setNumber(wake, "YMin", 0)
    field.setNumber(wake, "YMax", H)
    both = field.add("Min")
    field.setNumbers(both, "FieldsList", [near, wake])
    field.setAsBackgroundMesh(both)
    for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    msh, _, facets = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return msh, facets


def sampling_grid():
    xs = np.round(np.arange(0, L + 1e-9, GRID_STEP), 10)
    ys = np.round(np.arange(0, H + 1e-9, GRID_STEP), 10)
    X, Y = np.meshgrid(xs, ys, indexing="xy")              # (len(ys), len(xs))
    inside = (X - CX) ** 2 + (Y - CY) ** 2 <= R ** 2
    return xs, ys, X, Y, inside


class PointSampler:
    """Evaluates a vector P2 field at the fluid grid points (zero inside the cylinder)."""

    def __init__(self, msh):
        xs, ys, X, Y, inside = sampling_grid()
        self.shape, self.inside = X.shape, inside
        pts = np.zeros((X.size, 3))
        pts[:, 0], pts[:, 1] = X.ravel(), Y.ravel()
        fluid = np.flatnonzero(~inside.ravel())
        tree = geometry.bb_tree(msh, msh.topology.dim)
        colliding = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts[fluid]),
                                                     pts[fluid])
        cells = np.array([colliding.links(i)[0] if len(colliding.links(i)) else -1 for i in range(len(fluid))])
        if np.any(cells < 0):
            raise RuntimeError(f"{np.sum(cells < 0)} fluid grid points lie outside the mesh")
        self.index, self.points, self.cells = fluid, pts[fluid], cells
        self.npts = X.size

    def __call__(self, u):
        values = u.eval(self.points, self.cells)             # (n_fluid, 2)
        out = np.zeros((2, self.npts))
        out[:, self.index] = values.T
        return out.ravel()                                    # [u on grid, v on grid]


def ramp(t, t_ramp=0.5):
    return 1.0 if t >= t_ramp else 0.5 * (1 - np.cos(np.pi * t / t_ramp))


def solve(args):
    t_start = time.time()
    msh, facets = build_mesh(args.h_cyl, args.h_wake, args.h_far)
    V = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 2, shape=(2,)))
    Q = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1))
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    u_inlet = fem.Function(V)

    def set_inlet(t):
        scale = ramp(t)
        u_inlet.interpolate(lambda x: np.vstack((scale * 4 * U_MAX * x[1] * (H - x[1]) / H ** 2,
                                                 np.zeros(x.shape[1]))))

    set_inlet(0.0)
    zero = np.zeros(2, dtype=PETSc.ScalarType)
    bcu = [fem.dirichletbc(u_inlet, fem.locate_dofs_topological(V, fdim, facets.find(INLET))),
           fem.dirichletbc(zero, fem.locate_dofs_topological(V, fdim, facets.find(WALLS)), V),
           fem.dirichletbc(zero, fem.locate_dofs_topological(V, fdim, facets.find(CYLINDER)), V)]
    bcp = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, facets.find(OUTLET)), Q)]

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    u_, u_s, u_n, u_n1 = fem.Function(V), fem.Function(V), fem.Function(V), fem.Function(V)
    p_, phi = fem.Function(Q), fem.Function(Q)
    k = fem.Constant(msh, PETSc.ScalarType(args.dt))
    mu = fem.Constant(msh, PETSc.ScalarType(args.nu))
    dx = ufl.dx

    F1 = (ufl.dot(u - u_n, v) / k * dx
          + ufl.inner(ufl.dot(1.5 * u_n - 0.5 * u_n1, 0.5 * ufl.nabla_grad(u + u_n)), v) * dx
          + 0.5 * mu * ufl.inner(ufl.grad(u + u_n), ufl.grad(v)) * dx
          - ufl.dot(p_, ufl.div(v)) * dx)
    a1, L1 = fem.form(ufl.lhs(F1)), fem.form(ufl.rhs(F1))
    a2 = fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * dx)
    L2 = fem.form(-ufl.div(u_s) * q / k * dx)
    a3 = fem.form(ufl.dot(u, v) * dx)
    L3 = fem.form(ufl.dot(u_s, v) * dx - k * ufl.dot(ufl.grad(phi), v) * dx)

    A1 = create_matrix(a1)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    A3 = assemble_matrix(a3)
    A3.assemble()
    b1, b2, b3 = create_vector(L1), create_vector(L2), create_vector(L3)

    def ksp(A, kind, pc):
        s = PETSc.KSP().create(msh.comm)
        s.setOperators(A)
        s.setType(kind)
        s.getPC().setType(pc)
        if pc == "hypre":
            s.getPC().setHYPREType("boomeramg")
        s.setTolerances(rtol=1e-10, atol=1e-12)
        return s

    solver1, solver2, solver3 = ksp(A1, "bcgs", "jacobi"), ksp(A2, "minres", "hypre"), ksp(A3, "cg", "sor")

    n_cyl = -ufl.FacetNormal(msh)                            # out of the cylinder, into the fluid
    ds_cyl = ufl.Measure("ds", domain=msh, subdomain_data=facets, subdomain_id=CYLINDER)
    sigma = -p_ * ufl.Identity(2) + mu * (ufl.grad(u_) + ufl.grad(u_).T)
    coef = 2.0 / (U_MEAN ** 2 * D)
    drag = fem.form(coef * ufl.dot(sigma, n_cyl)[0] * ds_cyl)
    lift = fem.form(coef * ufl.dot(sigma, n_cyl)[1] * ds_cyl)

    sampler = PointSampler(msh)
    num_steps = int(round(args.T / args.dt))
    n_save = num_steps // args.save_every + 1
    snaps = np.zeros((n_save, 2 * sampler.npts), dtype=np.float32)
    t_save = np.zeros(n_save)
    cd, cl, t_steps = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
    snaps[0] = sampler(u_)
    j = 1
    t = 0.0
    print(f"Re = {U_MEAN * D / args.nu:g} (nu {args.nu:g})", flush=True)
    print(f"mesh: {msh.topology.index_map(2).size_global} cells, velocity dofs "
          f"{V.dofmap.index_map.size_global * 2}, pressure dofs {Q.dofmap.index_map.size_global}, "
          f"grid {sampler.shape}, steps {num_steps}", flush=True)
    for i in range(num_steps):
        t += args.dt
        if t <= 0.5 + args.dt:
            set_inlet(t)
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()
        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
        p_.x.scatter_forward()

        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        u_n1.x.array[:] = u_n.x.array
        u_n.x.array[:] = u_.x.array
        t_steps[i] = t
        cd[i] = msh.comm.allreduce(fem.assemble_scalar(drag), op=MPI.SUM)
        cl[i] = msh.comm.allreduce(fem.assemble_scalar(lift), op=MPI.SUM)
        if not np.isfinite(cd[i]):
            raise RuntimeError(f"non-finite drag at t = {t:.4f}")
        if (i + 1) % args.save_every == 0:
            snaps[j] = sampler(u_)
            t_save[j] = t
            j += 1
        if (i + 1) % max(1, num_steps // 40) == 0:
            print(f"  t = {t:7.3f}  Cd {cd[i]:.4f}  Cl {cl[i]:+.4f}  ({time.time() - t_start:.0f} s)", flush=True)
    xs, ys, _, _, inside = sampling_grid()
    np.savez(args.out, t=t_save[:j], snapshots=snaps[:j], t_steps=t_steps, cd=cd, cl=cl, xs=xs, ys=ys,
             inside=inside, dt=args.dt, nu=args.nu, h_cyl=args.h_cyl, h_wake=args.h_wake, h_far=args.h_far,
             cells=msh.topology.index_map(2).size_global, runtime=time.time() - t_start)
    print(f"saved {args.out}: {j} snapshots, {time.time() - t_start:.0f} s", flush=True)


def shedding_stats(t, cd, cl, t_from):
    """Benchmark quantities over t >= t_from: max Cd, max Cl, Strouhal number from the lift period."""
    keep = t >= t_from
    t, cd, cl = t[keep], cd[keep], cl[keep]
    up = np.flatnonzero((cl[:-1] < 0) & (cl[1:] >= 0))
    crossings = t[up] - cl[up] * (t[up + 1] - t[up]) / (cl[up + 1] - cl[up])
    period = np.mean(np.diff(crossings)) if len(crossings) > 2 else np.nan
    return dict(cd_max=float(cd.max()), cl_max=float(cl.max()), cl_min=float(cl.min()),
                strouhal=float(D / (U_MEAN * period)), period=float(period))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=6.25e-4)
    ap.add_argument("--save_every", type=int, default=8)
    ap.add_argument("--h_cyl", type=float, default=0.004)
    ap.add_argument("--h_wake", type=float, default=0.0125)
    ap.add_argument("--h_far", type=float, default=0.03)
    ap.add_argument("--nu", type=float, default=NU, help="kinematic viscosity; Re = U_MEAN * D / nu (default 100)")
    solve(ap.parse_args())
