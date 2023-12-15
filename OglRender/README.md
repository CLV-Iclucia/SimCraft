# Foundation of Computer Graphics Lab-1

## Dependencies:

- [GLFW](https://www.glfw.org/)

This is for cross-platform window management.

- [GLAD](https://glad.dav1d.de/)

This is for loading OpenGL functions.

- [GLM](https://www.opengl.org/sdk/libs/GLM/)

This is for 3D math toolkits.

- [Imgui](https://github.com/ocornut/imgui)

This is for GUI utils. Although it is also OK to use GLFW to implement interactions on my own (and I have done that before), I simply
want to make my life easier.

## How to Build demo

run 
```
mkdir build
cd build
cmake ..
make demo
```


### 0. First things first: warp things in C++

In `OglRender/include/ogl-render/ogl-ctx.h` I wrap up some essential OpenGL APIs and concepts in C++ and manage all the
resources using RAII. 

In `OglRender/include/ogl-render/shader-prog.h` I wrap up shader APIs in C++. `ShaderProg` class constructs a shader program
using specified shader paths automatically and manage all the uniform variables.

### 1. Draw a 2D computer

In `OglRender/apps/demo.cc` I implemented a class `DrawBoard` which stores all the primitives to render the image. And I
use the wrapped OpenGL API to render the image.

In this demo I only showed the process of drawing 2 rectangles and 1 triangle, but adding other primitives is similar to this.

### 2. Primitive selection and colour edition

To do this I implemented the `rayTriangleIntersection` function (which is adapted from my implementation in GAMES101 homework).

I shoot a ray from the coordinate of the cursor, and calculate the intersection point with the triangles/rectangles.

This enables us to figure out the primitive that the cursor is pointing at, and we can get all the information we need.