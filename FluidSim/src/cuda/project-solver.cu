//
// Created by creeper on 24-3-26.
//
#include <FluidSim/cuda/project-solver.h>
#include <cub/cub.cuh>
#include <format>

namespace fluid::cuda {
    enum Directions {
        Left,
        Right,
        Up,
        Down,
        Front,
        Back
    };

    static CUDA_GLOBAL void kernrelApplyCompressedMatrix(
            CudaSurfaceAccessor <Real> Adiag,
            std::array<CudaSurfaceAccessor<Real>, 6>
            Aneighbour,
            CudaSurfaceAccessor <Real> x,
            CudaSurfaceAccessor <uint8_t> active,
            CudaSurfaceAccessor <Real> b,
            int width, int height, int depth) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (!active.read(i, j, k)) return;
        Real t = Adiag.read(i, j, k) * x.read(i, j, k);
        t += active.read<cudaBoundaryModeZero>(i - 1, j, k) *
             Aneighbour[Left].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i - 1, j, k);
        t += active.read<cudaBoundaryModeZero>(i, j - 1, k) *
             Aneighbour[Down].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i, j - 1, k);
        t += active.read<cudaBoundaryModeZero>(i - 1, j, k) *
             Aneighbour[Back].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i, j, k - 1);
        t += active.read<cudaBoundaryModeZero>(i + 1, j, k) *
             Aneighbour[Right].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i + 1, j, k);
        t += active.read<cudaBoundaryModeZero>(i, j + 1, k) *
             Aneighbour[Up].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i, j + 1, k);
        t += active.read<cudaBoundaryModeZero>(i, j, k + 1) *
             Aneighbour[Front].read(i, j, k) *
             x.read<cudaBoundaryModeZero>(i, j, k + 1);
        b.write(t, i, j, k);
    }

    static CUDA_GLOBAL void kernelSaxpy(CudaSurfaceAccessor <Real> x,
                                        CudaSurfaceAccessor <Real> y,
                                        Real alpha,
                                        CudaSurfaceAccessor <uint8_t> active,
                                        int width, int height, int depth) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (!active.read(i, j, k)) return;
        x.write(x.read(i, j, k) + alpha * y.read(i, j, k), i, j, k);
    }

    static CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor <Real> x,
                                              CudaSurfaceAccessor <Real> y,
                                              Real alpha,
                                              CudaSurfaceAccessor <uint8_t> active,
                                              int width, int height, int depth) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (!active.read(i, j, k)) return;
        x.write(x.read(i, j, k) + alpha * y.read(i, j, k), i, j, k);
    }

    static CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<double> surfaceA,
                                             CudaSurfaceAccessor<double> surfaceB,
                                             CudaSurfaceAccessor <uint8_t> active,
                                             int3 dimensions, double *result) {
        get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
        double valueA = surfaceA.read(x, y, z);
        double valueB = surfaceB.read(x, y, z);
        double local_result = valueA * valueB * active.read(x, y, z);
        using BlockReduce = cub::BlockReduce<double, core::kThreadBlockSize3D,
                cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                core::kThreadBlockSize3D,
                core::kThreadBlockSize3D>;
        CUDA_SHARED
        BlockReduce::TempStorage temp_storage;
        double block_result{};
        BlockReduce(temp_storage).Sum(local_result,
                                      blockDim.x * blockDim.y * blockDim.z);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            atomicAdd(result, block_result);
        }
    }

    static CUDA_GLOBAL void kernelLinfNorm(CudaSurfaceAccessor <Real> surface,
                                           CudaSurfaceAccessor <uint8_t> active,
                                           int3 dimensions, Real *result) {
        get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
        Real value = surface.read(x, y, z);
        Real local_result = value * active.read(x, y, z);
        using BlockReduce = cub::BlockReduce<Real, core::kThreadBlockSize3D,
                cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                core::kThreadBlockSize3D,
                core::kThreadBlockSize3D>;
        CUDA_SHARED
        BlockReduce::TempStorage temp_storage;
        BlockReduce(temp_storage).Max(local_result,
                                      blockDim.x * blockDim.y * blockDim.z);
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            atomicMax(result, block_result);
    }

    static void saxpy(CudaSurfaceAccessor <Real> x,
                      CudaSurfaceAccessor <Real> y,
                      Real alpha,
                      CudaSurfaceAccessor <uint8_t> active,
                      int width, int height, int depth) {
        cudaSafeCall(kernelSaxpy<<<LAUNCH_THREADS_3D(width, height, depth)>>>(
                x, y, alpha, active, width, height, depth));
    }

    static void dotProduct(CudaSurfaceAccessor <Real> surfaceA,
                           CudaSurfaceAccessor <Real> surfaceB,
                           CudaSurfaceAccessor <uint8_t> active,
                           int width, int height, int depth, Real *result) {
        int3 dimensions = make_int3(width, height, depth);
        cudaSafeCall(kernelDotProduct<<<LAUNCH_THREADS_3D(width, height, depth)>>>(
                surfaceA, surfaceB, active, dimensions, result));
    }

    static CUDA_GLOBAL void kernelComputeAreaWeights(
            CudaSurfaceAccessor<double> uWeights,
            CudaSurfaceAccessor<double> vWeights,
            CudaSurfaceAccessor<double> wWeights,
            CudaSurfaceAccessor<double> fluidSdf,
            CudaSurfaceAccessor<double> colliderSdf,
            int width, int height, int depth, double h) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (i == 0 || i == uWeights.width() - 1) {
            uWeights(i, j, k) = 0.0;
            return;
        }
        if (fluidSdf.read(i - 1, j, k) < 0.0 || fluidSdf.read(i, j, k) < 0.0) {
            double3 p = make_double3(i * h, (j + 0.5) * h, (k + 0.5) * h);
            Real bu = colliderSdf.eval(p + make_double3(0.0, 1, -1) * 0.5 * h);
            Real bd = colliderSdf.eval(p + make_double3(0.0, -1, -1) * 0.5 * h);
            Real fd = colliderSdf.eval(p + make_double3(0.0, -1, 1) * 0.5 * h);
            Real fu = colliderSdf.eval(p + make_double3(0.0, 1, 1) * 0.5 * h);
            Real frac = fractionInside(bu, bd, fd, fu);
            uWeights(i, j, k) = 1.0 - frac;
            assert(notNan(uWeights(i, j, k)));
        }
        if (j == 0 || j == vWeights.height() - 1) {
            vWeights(i, j, k) = 0.0;
            return;
        }
        if (fluidSdf(i, j - 1, k) < 0.0 || fluidSdf(i, j, k) < 0.0) {
            double3 p = make_double3((i + 0.5) * h, j * h, (k + 0.5) * h);
            Real lb = colliderSdf.eval(p + make_double3(-1, 0.0, -1) * 0.5 * h);
            Real rb = colliderSdf.eval(p + make_double3(1, 0.0, -1) * 0.5 * h);
            Real rf = colliderSdf.eval(p + make_double3(1, 0.0, 1) * 0.5 * h);
            Real lf = colliderSdf.eval(p + make_double3(-1, 0.0, 1) * 0.5 * h);
            Real frac = fractionInside(lb, rb, rf, lf);
            assert(frac >= 0.0 && frac <= 1.0);
            vWeights(i, j, k) = 1.0 - frac;
            assert(notNan(vWeights(i, j, k)));
        }
        if (k == 0 || k == wWeights.depth() - 1) {
            wWeights(i, j, k) = 0.0;
            return;
        }
        if (fluidSdf(i, j, k - 1) < 0.0 || fluidSdf(i, j, k) < 0.0) {
            double3 p = make_double3((i + 0.5) * h, (j + 0.5) * h, k * h);
            Real ld = colliderSdf.eval(p + make_double3(-1, -1, 0.0) * 0.5 * h);
            Real lu = colliderSdf.eval(p + make_double3(-1, 1, 0.0) * 0.5 * h);
            Real ru = colliderSdf.eval(p + make_double3(1, 1, 0.0) * 0.5 * h);
            Real rd = colliderSdf.eval(p + make_double3(1, -1, 0.0) * 0.5 * h);
            Real frac = fractionInside(ld, lu, ru, rd);
            assert(frac >= 0.0 && frac <= 1.0);
            wWeights(i, j, k) = 1.0 - frac;
            assert(notNan(wWeights(i, j, k)));
        }
    }

    static CUDA_GLOBAL void kernelComputeMatrix(
            CudaSurfaceAccessor <Real> Adiag,
            CudaSurfaceAccessor <Real> Aneighbour[6],
            CudaSurfaceAccessor <Real> uWeights,
            CudaSurfaceAccessor <Real> vWeights,
            CudaSurfaceAccessor <Real> wWeights,
            CudaSurfaceAccessor <Real> rhs,
            CudaSurfaceAccessor <uint8_t> active,
            CudaSurfaceAccessor <Real> ug,
            CudaSurfaceAccessor <Real> vg,
            CudaSurfaceAccessor <Real> wg,
            CudaSurfaceAccessor <Real> fluidSdf,
            Real h,
            Real dt,
            int width, int height, int depth) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (uWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
            uWeights.read<cudaBoundaryModeZero>(i + 1, j, k) == 0.0 &&
            vWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
            vWeights.read<cudaBoundaryModeZero>(i, j + 1, k) == 0.0 &&
            wWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
            wWeights.read<cudaBoundaryModeZero>(i, j, k + 1) == 0.0)
            return;
        if (fluidSdf(i, j, k) > 0.0)
            return;
        active(i, j, k) = true;
        Real signed_dist = fluidSdf(i, j, k);
        Real factor = dt / h;
        assert(notNan(ug(i, j, k)));
        assert(notNan(vg(i, j, k)));
        assert(notNan(wg(i, j, k)));

        // left
        if (i > 0) {
            if (fluidSdf(i - 1, j, k) > 0.0) {
                Real theta = fmin(
                        fluidSdf(i - 1, j, k) / (fluidSdf(i - 1, j, k) - signed_dist),
                        0.99);
                Adiag(i, j, k) += uWeights(i, j, k) * factor / (1.0 - theta);
            } else {
                Adiag(i, j, k) += uWeights(i, j, k) * factor;
                Aneighbour[Left](i, j, k) -= uWeights(i, j, k) * factor;
            }
            rhs(i, j, k) += uWeights(i, j, k) * ug(i, j, k);
        }

        // right
        if (i < fluidSdf.width() - 1) {
            if (fluidSdf(i + 1, j, k) > 0.0) {
                Real theta = fmax(
                        signed_dist / (signed_dist - fluidSdf(i + 1, j, k)), 0.01);
                Adiag(i, j, k) += uWeights(i + 1, j, k) * factor / theta;
            } else {
                if (i < fluidSdf.width() - 1) {
                    Adiag(i, j, k) += uWeights(i + 1, j, k) * factor;
                    Aneighbour[Right](i, j, k) -= uWeights(i + 1, j, k) * factor;
                }
            }
            rhs(i, j, k) -= uWeights(i + 1, j, k) * ug(i + 1, j, k);
        }

        // down
        if (j > 0) {
            if (fluidSdf(i, j - 1, k) > 0.0) {
                Real theta = fmin(
                        fluidSdf(i, j - 1, k) / (fluidSdf(i, j - 1, k) - signed_dist),
                        0.99);
                Adiag(i, j, k) += vWeights(i, j, k) * factor / (1.0 - theta);
            } else {
                Adiag(i, j, k) += vWeights(i, j, k) * factor;
                Aneighbour[Down](i, j, k) -= vWeights(i, j, k) * factor;
            }
            rhs(i, j, k) += vWeights(i, j, k) * vg(i, j, k);
        }
        // up
        if (j < fluidSdf.width() - 1) {
            if (fluidSdf(i, j + 1, k) > 0.0) {
                Real theta = fmax(
                        signed_dist / (signed_dist - fluidSdf(i, j + 1, k)), 0.01);
                Adiag(i, j, k) += vWeights(i, j + 1, k) * factor / theta;
            } else {
                Adiag(i, j, k) += vWeights(i, j + 1, k) * factor;
                Aneighbour[Up](i, j, k) -= vWeights(i, j + 1, k) * factor;
            }
            rhs(i, j, k) -= vWeights(i, j + 1, k) * vg(i, j + 1, k);
        }

        // back
        if (k > 0) {
            if (fluidSdf(i, j, k - 1) > 0.0) {
                Real theta = fmin(
                        fluidSdf(i, j, k - 1) / (fluidSdf(i, j, k - 1) - signed_dist),
                        0.99);
                Adiag(i, j, k) += wWeights(i, j, k) * factor / (1.0 - theta);
            } else {
                Adiag(i, j, k) += wWeights(i, j, k) * factor;
                Aneighbour[Back](i, j, k) -= wWeights(i, j, k) * factor;
            }
            rhs(i, j, k) += wWeights(i, j, k) * wg(i, j, k);
        }

        // front
        if (k < fluidSdf.depth() - 1) {
            if (fluidSdf(i, j, k + 1) > 0.0) {
                Real theta = fmax(
                        signed_dist / (signed_dist - fluidSdf(i, j, k + 1)), 0.01);
                Adiag(i, j, k) += wWeights(i, j, k + 1) * factor / theta;
            } else {
                Adiag(i, j, k) += wWeights(i, j, k + 1) * factor;
                Aneighbour[Front](i, j, k) -= wWeights(i, j, k + 1) * factor;
            }
            rhs(i, j, k) -= wWeights(i, j, k + 1) * wg(i, j, k + 1);
        }
        assert(Adiag(i, j, k) > 0.0);
        assert(notNan(rhs(i, j, k)));
    }

    static void LinfNorm(CudaSurfaceAccessor <Real> surface,
                         CudaSurfaceAccessor <uint8_t> active,
                         int width, int height, int depth, Real *result) {
        int3 dimensions = make_int3(width, height, depth);
        cudaSafeCall(kernelLinfNorm<<<LAUNCH_THREADS_3D(width, height, depth)>>>(
                surface, active, dimensions, result));
    }

    static CUDA_GLOBAL void kernelProjectVelocity(
            CudaSurfaceAccessor <Real> ug,
            CudaSurfaceAccessor <Real> vg,
            CudaSurfaceAccessor <Real> wg,
            CudaSurfaceAccessor <Real> pg,
            CudaSurfaceAccessor <Real> fluidSdf,
            CudaSurfaceAccessor <Real> colliderSdf,
            Real h,
            Real dt,
            int width, int height, int depth) {
        get_and_restrict_tid_3d(i, j, k, width, height, depth);
        if (uWeights(i, j, k) <= 0.0) return;
        if (i == 0 || i == ug.width() - 1) {
            ug(i, j, k) = 0.0;
            return;
        }
        Real sd_left = fluid_sdf(i - 1, j, k);
        Real sd_right = fluid_sdf(i, j, k);
        assert(notNan(pg(i, j, k)));
        if (sd_left >= 0.0 && sd_right >= 0.0) return;
        if (sd_left < 0.0 && sd_right < 0.0) {
            ug(i, j, k) -= (pg(i, j, k) - pg(i - 1, j, k)) * dt / h;
            return;
        }
        if (sd_left < 0.0) {
            Real theta = fmax(sd_left / (sd_left - sd_right), 0.01);
            ug(i, j, k) += pg(i - 1, j, k) * dt / h / theta;
        } else {
            Real theta = fmin(sd_left / (sd_left - sd_right), 0.99);
            ug(i, j, k) -= pg(i, j, k) * dt / h / (1.0 - theta);
        }
        if (vWeights(i, j, k) <= 0.0) return;
        if (j == 0 || j == vg.height() - 1) {
            vg(i, j, k) = 0.0;
            return;
        }
        Real sd_down = fluid_sdf(i, j - 1, k);
        Real sd_up = fluid_sdf(i, j, k);
        assert(notNan(pg(i, j, k)));
        if (sd_down >= 0.0 && sd_up >= 0.0) return;
        if (sd_down < 0.0 && sd_up < 0.0) {
            vg(i, j, k) -= (pg(i, j, k) - pg(i, j - 1, k)) * dt / h;
            return;
        }
        if (sd_down < 0.0) {
            Real theta = fmax(sd_down / (sd_down - sd_up), 0.01);
            vg(i, j, k) += pg(i, j - 1, k) * dt / h / theta;
        } else {
            Real theta = fmin(sd_down / (sd_down - sd_up), 0.99);
            vg(i, j, k) -= pg(i, j, k) * dt / h / (1.0 - theta);
        }
        if (wWeights(i, j, k) <= 0.0) return;
        if (k == 0 || k == wg.depth() - 1) {
            wg(i, j, k) = 0.0;
            return;
        }
        assert(notNan(pg(i, j, k)));
        Real sd_back = fluid_sdf(i, j, k - 1);
        Real sd_front = fluid_sdf(i, j, k);
        if (sd_back >= 0.0 && sd_front >= 0.0) return;
        if (sd_back < 0.0 && sd_front < 0.0) {
            wg(i, j, k) -= (pg(i, j, k) - pg(i, j, k - 1)) * dt / h;
            return;
        }
        if (sd_back < 0.0) {
            Real theta = fmax(sd_back / (sd_back - sd_front), 0.01);
            wg(i, j, k) += pg(i, j, k - 1) * dt / h / theta;
        } else {
            Real theta = fmin(sd_back / (sd_back - sd_front), 0.99);
            wg(i, j, k) -= pg(i, j, k) * dt / h / (1.0 - theta);
        }
    }

    void FvmSolver::buildSystem(CudaSurfaceAccessor <Real> ug,
                                CudaSurfaceAccessor <Real> vg,
                                CudaSurfaceAccessor <Real> wg,
                                CudaSurfaceAccessor <Real> fluidSdf,
                                CudaSurfaceAccessor <Real> colliderSdf,
                                Real dt) {
        Real h = ug.gridSpacing().x;
        Adiag.fill(0);
        Aneighbour[Left].fill(0);
        Aneighbour[Right].fill(0);
        Aneighbour[Up].fill(0);
        Aneighbour[Down].fill(0);
        Aneighbour[Front].fill(0);
        Aneighbour[Back].fill(0);
        uWeights.fill(0);
        vWeights.fill(0);
        wWeights.fill(0);
        rhs.fill(0);
        active.fill(false);
    }

    Real CgSolver::solve(CudaSurfaceAccessor <Real> Adiag,
                         const std::array<CudaSurfaceAccessor<Real>, 6> &Aneighbour,
                         CudaSurfaceAccessor <Real> rhs,
                         CudaSurfaceAccessor <uint8_t> active,
                         CudaSurfaceAccessor <Real> pg) {
        pg.fill(0);
        r.copyFrom(rhs);
        Real residual = LinfNorm(r, active);
        if (residual < tolerance) {
            std::cout << "naturally converged" << std::endl;
            return residual;
        }
        if (preconditioner)
            preconditioner->precond(Adiag, Aneighbour, rhs, active, z);
        else
            z.copyFrom(r);
        s.copyFrom(z);
        Real sigma = dotProduct(z, r, active);
        int iter = 1;
        for (; iter < max_iterations; iter++) {
            applyCompressedMatrix(Adiag, Aneighbour, s, active, z);
            Real sdotz = dotProduct(s, z, active);
            assert(sdotz != 0);
            Real alpha = sigma / sdotz;
            saxpy(pg, s, alpha, active);
            saxpy(r, z, -alpha, active);
            residual = LinfNorm(r, active);
            if (residual < tolerance) break;
            if (preconditioner)
                preconditioner->precond(Adiag, Aneighbour, r, active, z);
            Real sigma_new = dotProduct(z, r, active);
            assert(sigma != 0);
            Real beta = sigma_new / sigma;
            scaleAndAdd(s, z, beta, active);
            sigma = sigma_new;
        }
        std::cout << std::format("PCG iterations: {}", iter) << std::endl;
        return residual;
    }

    void FvmSolver3D::project(FaceCentredGrid<Real, Real, 3, 0> &ug,
                              FaceCentredGrid<Real, Real, 3, 1> &vg,
                              FaceCentredGrid<Real, Real, 3, 2> &wg,
                              const Array3D <Real> &pg,
                              const SDF<3> &fluid_sdf,
                              const SDF<3> &collider_sdf,
                              Real dt) {
        Real h = ug.gridSpacing().x;
    }

    Real CgSolver::solve(CudaSurfaceAccessor <Real> Adiag,
                         const std::array<CudaSurfaceAccessor<Real>, 6> &Aneighbour,
                         CudaSurfaceAccessor <Real> rhs,
                         CudaSurfaceAccessor <uint8_t> active,
                         CudaSurfaceAccessor <Real> pg) {
        pg.fill(0);
        r.copyFrom(rhs);
        Real residual = LinfNorm(r, active);
        if (residual < tolerance) {
            std::cout << "naturally converged" << std::endl;
            return residual;
        }
        if (preconditioner)
            preconditioner->precond(Adiag, Aneighbour, rhs, active, z);
        else
            z.copyFrom(r);
        s.copyFrom(z);
        Real sigma = dotProduct(z, r, active);
        int iter = 1;
        for (; iter < max_iterations; iter++) {
            applyCompressedMatrix(Adiag, Aneighbour, s, active, z);
            Real sdotz = dotProduct(s, z, active);
            assert(sdotz != 0);
            Real alpha = sigma / sdotz;
            saxpy(pg, s->surfAccessor(), alpha, active, width, height, depth);
            saxpy(r, z, -alpha, active, width, height, depth);
            Linf(r, active, dev_ptr_residual);
            if (residual < tolerance) break;
            if (preconditioner)
                preconditioner->precond(Adiag, Aneighbour, r, active, z);
            dotProduct(z, r, active, width, height, depth, dev_ptr_sigma);
            assert(sigma != 0);
            Real beta = sigma_new / sigma;
            scaleAndAdd(s, z, beta, active, width, height, depth);
            sigma = sigma_new;
        }
        std::cout << std::format("PCG iterations: {}", iter) << std::endl;
        return residual;
    }
}