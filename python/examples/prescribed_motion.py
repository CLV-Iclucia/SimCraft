"""
SimCraft Example: Prescribed Motion
====================================

A vertex is driven by a Python lambda — sinusoidal oscillation in x.
Demonstrates: prescribe_motion with a Python callable.

Usage:
    set PYTHONPATH=<build-dir>
    python examples/prescribed_motion.py
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
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

material = simcraft.NeoHookean(young=2e5, poisson=0.45)

system = simcraft.System()
system.add_elastic_body(mesh, material, density=800.0)
system.gravity = np.array([0.0, -9.81, 0.0])

# Pin bottom vertices
system.constraints.pin_vertices(np.array([0, 1, 2, 3], dtype=np.int32))

# Prescribe sinusoidal motion on vertex 10
system.constraints.prescribe_motion(10, lambda t: np.array([0.5 * np.sin(2 * np.pi * t), 0.0, 0.0]))

# ─── Run ─────────────────────────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=1e-3, kappa=1e8)
sim = simcraft.Simulation(system, integrator)
renderer = simcraft.Renderer(width=1280, height=720, title="Prescribed Motion")

print("Running... close window to stop.")
simcraft.run_and_display(sim, renderer, dt=0.01, steps=300)
print(f"Done. {sim.steps_completed} steps.")
