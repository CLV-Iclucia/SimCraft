"""Minimal ABD rigid body scene.

This uses the Runtime/ABD scene path rather than the legacy FEM System API.
"""
import numpy as np

import kiseki


vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

triangles = np.array([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
], dtype=np.int32)

initial_transform = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
], dtype=np.float64)

body = kiseki.ABDRigidBody(
    vertices=vertices,
    triangles=triangles,
    initial_transform=initial_transform,
    density=1000.0,
    contact_thickness=1.0e-3,
)

scene = kiseki.ABDScene()
scene.time_step = 1.0 / 60.0
scene.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float64)
scene.solver.enable_contact = True
scene.solver.contact_barrier_distance = 1.0e-3
scene.solver.contact_stiffness = 1.0e5
scene.add_body(body, tag="falling-abd-tetra")

sim = scene.build()
print("initial body positions:")
print(sim.body_positions(0))

sim.step()
print("after one step:")
print(sim.body_positions(0))
