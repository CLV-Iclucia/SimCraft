// add header protection
#ifndef POISSONSOLVER_UTIL_H
#define POISSONSOLVER_UTIL_H

namespace poisson {
void applyLaplacian(double* u, double* f, int n, int m);
void zeroFill(double* u, int n, int m);
void rbGaussSeidel(double* u, const double *f, int n, int m, int color);
void rbGaussSeidelReverse(double* u, const double *f, int n, int m, int color);
void gaussSeidel(double *u, const double *f, int n, int m, int i, int j);
void saxpy(double* y, double a, double* x, int n, int m);
void scaleAndAdd(double* y, double a, double* x, int n, int m);
double normLinf(double* u, int n, int m);
double normSqr(double* u, int n, int m);
void computeResidual(double* u, double* f, double* r, int n, int m);
double computeNullSpace(double* r, int n, int m);
double dot(double* u, double* v, int n, int m);
void scaleDiv(double* u, double scale, int n, int m);
void scaleMul(double* u, double scale, int n, int m);
}

#endif