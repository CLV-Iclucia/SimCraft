#include <algorithm>
#include <cstring>
namespace poisson {
void zeroFill(double *u, int n, int m) {
  memset(u, 0, sizeof(double) * n * m);
}

void computeResidual(double *u, double *f, double *r, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      double v = (i > 0 ? u[i * m + j] - u[(i - 1) * m + j] : 0.f) +
          (i < n - 1 ? u[i * m + j] - u[(i + 1) * m + j] : 0.f) +
          (j > 0 ? u[i * m + j] - u[i * m + j - 1] : 0.f) +
          (j < m - 1 ? u[i * m + j] - u[i * m + j + 1] : 0.f);
      r[i * m + j] = f[i * m + j] - v;
    }
  }
}

double dot(double *u, double *v, int n, int m) {
  double res = 0.0;
  for (int i = 0; i < n * m; i++)
    res += static_cast<double>(u[i]) * static_cast<double>(v[i]);
  return static_cast<double>(res);
}

void rbGaussSeidel(double *u, double *f, int n, int m, int color) {
  for (int i = 0; i < n; i++) {
    for (int j = color ^ (i & 1); j < m; j += 2) {
      int cnt = 0;
      double v = 0;
      if (i > 0) {
        v += u[(i - 1) * m + j];
        cnt++;
      }
      if (i < n - 1) {
        v += u[(i + 1) * m + j];
        cnt++;
      }
      if (j > 0) {
        v += u[i * m + j - 1];
        cnt++;
      }
      if (j < m - 1) {
        v += u[i * m + j + 1];
        cnt++;
      }
      u[i * m + j] = (f[i * m + j] + v) / cnt;
    }
  }
}

// perf history: using block size of 4, 1.5 faster than the trivial loop
// perf history: only blocking y, even slower?
void applyLaplacian(double *output, double *input, int n, int m) {
  int B = 4;
  int nB = n / B;
  int mB = m / B;
  for (int iB = 0; iB < nB; iB++) {
    for (int jB = 0; jB < mB; jB++) {
      for (int ii = 0; ii < B; ii++) {
        for (int jj = 0; jj < B; jj++) {
          int i = iB * B + ii;
          int j = jB * B + jj;
          output[i * m + j] = (i > 0 ? input[i * m + j] - input[(i - 1) * m + j] : 0) +
              (i < n - 1 ? input[i * m + j] - input[(i + 1) * m + j] : 0) +
              (j > 0 ? input[i * m + j] - input[i * m + j - 1] : 0) +
              (j < m - 1 ? input[i * m + j] - input[i * m + j + 1] : 0);
        }
      }
    }
  }

}

double computeNullSpace(double *r, int n, int m) {
  double mu = 0;
  for (int i = 0; i < n * m; i++) mu += r[i];
  mu /= n * m;
  return mu;
}

void saxpy(double *y, double a, double *x, int n, int m) {
  for (int i = 0; i < n * m; i++) y[i] += a * x[i];
}

void scaleAndAdd(double *y, double a, double *x, int n, int m) {
  for (int i = 0; i < n * m; i++) y[i] = a * y[i] + x[i];
}

void scaleMul(double *v, double scale, int n, int m) {
  for (int i = 0; i < n * m; i++) v[i] *= scale;
}

void scaleDiv(double *v, double scale, int n, int m) {
  for (int i = 0; i < n * m; i++) v[i] /= scale;
}

double normLinf(double *u, int n, int m) {
  double ret = 0;
  for (int i = 0; i < n * m; i++)
    ret = std::max(ret, std::abs(u[i]));
  return ret;
}

double normSqr(double *u, int n, int m) {
  double ret = 0;
  for (int i = 0; i < n * m; i++)
    ret += u[i] * u[i];
  return ret;
}

}