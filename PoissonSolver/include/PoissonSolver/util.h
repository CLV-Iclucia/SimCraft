// add header protection
#ifndef POISSONSOLVER_UTIL_H
#define POISSONSOLVER_UTIL_H

namespace poisson {
void applyLaplacian(float* u, float* f, int n, int m);
void zeroFill(float* u, int n, int m);
void rbGaussSeidel(float* u, float* f, int n, int m, int color);
void saxpy(float* y, float a, float* x, int n, int m);
void scaleAndAdd(float* y, float a, float* x, float b, int n, int m);
}

#endif