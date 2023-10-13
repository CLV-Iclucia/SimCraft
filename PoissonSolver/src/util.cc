#include <PoissonSolver/poisson-solver.h>
#include <cstring>

namespace poisson {
void zeroFill(float* u, int n, int m) {
  memset(u, 0, sizeof(float) * n * m);
}

void rbGaussSeidel(float* u, float* f, int n, int m, int color) {
  for (int i = 0; i < n; i++) {
    for (int j = (i + color) & 1; j < m; j += 2) {
      float v = 0;
      if (i > 0) v += u[(i - 1) * m + j];
      if (i < n - 1) v += u[(i + 1) * m + j];
      if (j > 0) v += u[i * m + j - 1];
      if (j < m - 1) v += u[i * m + j + 1];
      v = (f[i * m + j] - v) / 4;
      u[i * m + j] += v;
    }
  }
}

void applyLaplacian(float* output, float* input, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      output[i * m + j] = 4 * input[i * m + j] -
                          (i > 0 ? input[(i - 1) * m + j] : 0) -
                          (i < n - 1 ? input[(i + 1) * m + j] : 0) -
                          (j > 0 ? input[i * m + j - 1] : 0) -
                          (j < m - 1 ? input[i * m + j + 1] : 0);
  }
}

void saxpy(float* y, float a, float* x, int n, int m) {
  for (int i = 0; i < n * m; i++) y[i] += a * x[i];
}

void scaleAndAdd(float* y, float a, float* x, float b, int n, int m) {
  for (int i = 0; i < n * m; i++) y[i] = a * y[i] + b * x[i];
}

}