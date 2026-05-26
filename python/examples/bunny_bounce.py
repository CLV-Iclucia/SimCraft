"""
SimCraft Example: Bunny Bounce
==============================

Stanford bunny with pinned vertices, dropping onto a rising ground plane.
Demonstrates: pin_vertices + kinematic body with constant velocity.

Usage:
    set PYTHONPATH=<build-dir>
    python examples/bunny_bounce.py
"""
import numpy as np

try:
    import simcraft
except ImportError:
    raise ImportError(
        "Cannot import simcraft. Install first:\n"
        "  python dev_setup.py                     (developer: after CLion build)\n"
        "  pip install .                            (user: from VS Dev Prompt)\n"
        "See python/README.md for details."
    )

from pathlib import Path

print(f"simcraft {simcraft.__version__}")

# ─── Setup ───────────────────────────────────────────────────────────────────
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "bunny.tobj"
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Bunny: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

material = simcraft.NeoHookean(young=5e5, poisson=0.4)

system = simcraft.System()
system.add_elastic_body(mesh, material, density=1000.0)
system.gravity = np.array([0.0, -9.81, 0.0])

# Pin top vertices
system.constraints.pin_vertices(np.array([0, 1, 2], dtype=np.int32))

# Rising ground plane
ground = simcraft.KinematicBody.plane(normal=np.array([0.0, 1.0, 0.0]), offset=-2.0)
ground.set_constant_velocity(np.array([0.0, 0.5, 0.0]))
system.add_kinematic_body(ground)

# ─── Run ─────────────────────────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=1e-3, kappa=1e9)
sim = simcraft.Simulation(system, integrator)
renderer = simcraft.Renderer(width=1280, height=720, title="Bunny Bounce")

print("Running... close window to stop.")
simcraft.run_and_display(sim, renderer, dt=0.005, steps=1000)
print(f"Done. {sim.steps_completed} steps.")
