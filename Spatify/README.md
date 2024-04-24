# Spatify: Spatial data structures can simplify my life!

## What is this

Spatify is a header-only library of various spatial data structures for computer graphics.

Now it includes:

- 2D and 3D arrays with optional paddings
- 2D and 3D grids, with different configurations that can be customized by template arguments
- Neighbour searching toolkits

To be added:
- Signed distance field (Now developing in other repos)
- Particle system
- Reconstruction toolkits (Also in other repos)
- BVH (Also in other repos)
- ...

## How to use it

Simply add it to your CMake project and include the headers you want! Remember to set the glm path!

## Dependencies

I use TBB for parallel execution and glm for vector calculations.