#include <DER/cpu/der-cpu-backend.h>

#include <DER/der-subsystem.h>

#include <Contact/contact-barrier.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::der
{
    namespace
    {
        static_assert(sizeof(RodDof) == 4 * sizeof(double));

        void requireCPU(runtime::ConstDofView view, const char* operation)
        {
            if (!view.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU DofView");
            }
        }

        void requireCPU(runtime::DofView view, const char* operation)
        {
            requireCPU(view.asConst(), operation);
        }

        void requireCPU(runtime::ConstGeometryView view, const char* operation)
        {
            if (!view.isCPU())
            {
                throw std::runtime_error(std::string(operation) +
                    " requires a CPU GeometryView");
            }
        }

        void requireCPU(runtime::GeometryView view, const char* operation)
        {
            requireCPU(view.asConst(), operation);
        }

        void requireDofCount(runtime::ConstDofView view,
                             int expected,
                             const char* operation)
        {
            if (view.scalarCount() < expected)
            {
                throw std::invalid_argument(std::string(operation) +
                    " received a DOF view that is too small");
            }
        }

        double vectorInfNorm(const Eigen::VectorXd& vector)
        {
            return vector.size() == 0 ? 0.0 : vector.cwiseAbs().maxCoeff();
        }

        int maxAbsIndex(const Eigen::VectorXd& vector)
        {
            if (vector.size() == 0)
            {
                return -1;
            }
            Eigen::Index index = 0;
            vector.cwiseAbs().maxCoeff(&index);
            return static_cast<int>(index);
        }

        struct ScalarLocation
        {
            int rod = -1;
            int vertex = -1;
            int lane = -1;
        };

        ScalarLocation locateScalar(const DERSubsystem& subsystem,
                                    int local_scalar)
        {
            const auto& offsets = subsystem.rodOffsets();
            const auto& rods = subsystem.rods();
            for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
                 ++rod_index)
            {
                const int base = offsets[static_cast<size_t>(rod_index)].q;
                const int scalar_count =
                    static_cast<int>(rods[static_cast<size_t>(rod_index)]
                                         .state()
                                         .size()) *
                    4;
                if (local_scalar >= base && local_scalar < base + scalar_count)
                {
                    const int relative = local_scalar - base;
                    return ScalarLocation{
                        .rod = rod_index,
                        .vertex = relative / 4,
                        .lane = relative % 4,
                    };
                }
            }
            return {};
        }

        void dumpDenseVector(const Eigen::VectorXd& vector,
                             const char* path)
        {
            std::ofstream out(path);
            if (!out)
            {
                return;
            }

            out << "# DER CPU backend vector\n";
            out << "size " << vector.size() << '\n';
            out << std::scientific << std::setprecision(17);
            for (int i = 0; i < vector.size(); ++i)
            {
                out << vector[i] << '\n';
            }
        }

        void dumpDenseMatrix(const Eigen::SparseMatrix<double>& matrix,
                             const char* path);

        void diagnoseDenseAssemblyMatrix(const char* label,
                                         const Eigen::MatrixXd& matrix,
                                         int subsystem_id,
                                         int rod_index,
                                         bool require_spd)
        {
            if (matrix.rows() == 0)
            {
                return;
            }
            if (!matrix.allFinite())
            {
                spdlog::warn(
                    "DER assembly matrix contains non-finite entries: "
                    "label={} subsystem={} rod={} scalars={}",
                    label,
                    subsystem_id,
                    rod_index,
                    matrix.rows());
                return;
            }

            const Eigen::MatrixXd skew = matrix - matrix.transpose();
            const double asymmetry = skew.cwiseAbs().maxCoeff();
            const Eigen::VectorXd diagonal = matrix.diagonal();
            const double min_diag = diagonal.minCoeff();
            const double max_diag = diagonal.maxCoeff();
            const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(symmetric);
            if (eigensolver.info() != Eigen::Success)
            {
                spdlog::warn(
                    "DER assembly matrix eigen analysis failed: label={} "
                    "subsystem={} rod={} scalars={} asymmetry={} min_diag={} "
                    "max_diag={}",
                    label,
                    subsystem_id,
                    rod_index,
                    matrix.rows(),
                    asymmetry,
                    min_diag,
                    max_diag);
                return;
            }

            const double min_eigen = eigensolver.eigenvalues().minCoeff();
            const double max_abs_eigen =
                eigensolver.eigenvalues().cwiseAbs().maxCoeff();
            const double tolerance =
                1.0e-10 * std::max(1.0, max_abs_eigen);
            const bool valid = require_spd ? min_eigen > tolerance
                                           : min_eigen >= -tolerance;
            if (valid && asymmetry <= 1.0e-8)
            {
                return;
            }

            spdlog::warn(
                "DER assembly matrix is not {}: label={} subsystem={} "
                "rod={} scalars={} min_eigen={} max_abs_eigen={} "
                "tolerance={} asymmetry={} min_diag={} max_diag={}",
                require_spd ? "SPD" : "PSD",
                label,
                subsystem_id,
                rod_index,
                matrix.rows(),
                min_eigen,
                max_abs_eigen,
                tolerance,
                asymmetry,
                min_diag,
                max_diag);
        }

        void diagnoseLocalMatrixSPD(const Eigen::SparseMatrix<double>& matrix,
                                    int subsystem_id,
                                    int internal_contact_count)
        {
            if (matrix.rows() == 0)
            {
                return;
            }

            Eigen::MatrixXd dense = Eigen::MatrixXd(matrix);
            if (!dense.allFinite())
            {
                spdlog::warn(
                    "DER local operator contains non-finite entries: "
                    "subsystem={} scalars={} internal_contacts={}",
                    subsystem_id,
                    matrix.rows(),
                    internal_contact_count);
                dumpDenseMatrix(matrix, "der-local-matrix-nonfinite.txt");
                return;
            }

            const Eigen::MatrixXd skew = dense - dense.transpose();
            const double asymmetry = skew.cwiseAbs().maxCoeff();
            const Eigen::VectorXd diagonal = dense.diagonal();
            const double min_diag = diagonal.minCoeff();
            const double max_diag = diagonal.maxCoeff();
            const Eigen::MatrixXd symmetric = 0.5 * (dense + dense.transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(symmetric);
            if (eigensolver.info() != Eigen::Success)
            {
                spdlog::warn(
                    "DER local operator eigen analysis failed: subsystem={} "
                    "scalars={} internal_contacts={} asymmetry={} "
                    "min_diag={} max_diag={}",
                    subsystem_id,
                    matrix.rows(),
                    internal_contact_count,
                    asymmetry,
                    min_diag,
                    max_diag);
                dumpDenseMatrix(matrix, "der-local-matrix-eigen-failure.txt");
                return;
            }

            const double min_eigen = eigensolver.eigenvalues().minCoeff();
            const double max_abs_eigen =
                eigensolver.eigenvalues().cwiseAbs().maxCoeff();
            if (min_eigen > 0.0 && asymmetry <= 1.0e-8 && min_diag > 0.0)
            {
                return;
            }

            spdlog::warn(
                "DER local operator is not SPD: subsystem={} scalars={} "
                "min_eigen={} max_abs_eigen={} asymmetry={} min_diag={} "
                "max_diag={} internal_contacts={}",
                subsystem_id,
                matrix.rows(),
                min_eigen,
                max_abs_eigen,
                asymmetry,
                min_diag,
                max_diag,
                internal_contact_count);
            dumpDenseMatrix(matrix, "der-local-matrix-not-spd.txt");
        }

        int contactPointCount(runtime::ContactCase type)
        {
            switch (type)
            {
            case runtime::ContactCase::PP:
                return 2;
            case runtime::ContactCase::PE:
                return 3;
            case runtime::ContactCase::PT:
            case runtime::ContactCase::EE:
                return 4;
            }
            return 0;
        }

        engine::contact::EBarrierStencil barrierStencilType(
            runtime::ContactCase type)
        {
            switch (type)
            {
            case runtime::ContactCase::PP:
                return engine::contact::EBarrierStencil::PP;
            case runtime::ContactCase::PE:
                return engine::contact::EBarrierStencil::PE;
            case runtime::ContactCase::PT:
                return engine::contact::EBarrierStencil::PT;
            case runtime::ContactCase::EE:
                return engine::contact::EBarrierStencil::EE;
            }
            return engine::contact::EBarrierStencil::PP;
        }

        int findSampleIndex(const DERSubsystem& subsystem, runtime::PointIdx point)
        {
            const auto& points = subsystem.geometryPointIds();
            const auto it = std::find(points.begin(), points.end(), point);
            if (it == points.end())
            {
                return -1;
            }
            return static_cast<int>(it - points.begin());
        }

        struct LocalContactStencil
        {
            std::array<int, 4> qOffsets{-1, -1, -1, -1};
            std::array<glm::dvec3, 4> x{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            std::array<glm::dvec3, 4> restX{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            int count = 0;
            double dHat = 0.0;
            double surfaceOffset = 0.0;
            double kappa = 0.0;
        };

        std::optional<LocalContactStencil> gatherLocalContact(
            const DERSubsystem& subsystem,
            const runtime::ContactStencil& contact)
        {
            LocalContactStencil local;
            local.count = contactPointCount(contact.type);
            local.dHat = contact.dHat;
            local.surfaceOffset = contact.thickness;
            local.kappa = contact.stiffness;
            if (local.count <= 0 || local.dHat <= 0.0 || local.kappa <= 0.0)
            {
                return std::nullopt;
            }

            const auto& samples = subsystem.geometrySamples();
            const auto& rods = subsystem.rods();
            for (int i = 0; i < local.count; ++i)
            {
                const int sample_index =
                    findSampleIndex(subsystem, contact.geometryIds[i]);
                if (sample_index < 0)
                {
                    return std::nullopt;
                }
                const DERGeometrySample& sample = samples[sample_index];
                const Rod& rod = rods[sample.rod];
                local.qOffsets[i] = sample.qOffset;
                local.x[i] = rod.state().position(sample.vertex);
                local.restX[i] = glm::dvec3(
                    rod.restState().blocks.at(sample.vertex));
            }
            return local;
        }

        engine::contact::GIPCBarrierStencil makeBarrierStencil(
            const LocalContactStencil& contact,
            runtime::ContactCase type)
        {
            engine::contact::GIPCBarrierStencil barrier;
            barrier.type = barrierStencilType(type);
            barrier.x = contact.x;
            barrier.restX = contact.restX;
            barrier.dHat = contact.dHat;
            barrier.reservedDist = contact.surfaceOffset;
            barrier.kappa = contact.kappa;
            return barrier;
        }

        void scatterBarrierValues(
            const LocalContactStencil& contact,
            const std::array<glm::dvec3, 4>& values,
            runtime::DofView y)
        {
            for (int i = 0; i < contact.count; i++)
            {
                const int offset = contact.qOffsets[i];
                y[offset + 0] += values[i].x;
                y[offset + 1] += values[i].y;
                y[offset + 2] += values[i].z;
            }
        }

        std::array<glm::dvec3, 4> gatherBarrierDirection(
            const LocalContactStencil& contact,
            runtime::ConstDofView dq)
        {
            std::array<glm::dvec3, 4> direction{
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0),
                glm::dvec3(0.0)
            };
            for (int i = 0; i < contact.count; i++)
            {
                const int offset = contact.qOffsets[i];
                direction[i] = glm::dvec3(
                    dq[offset + 0], dq[offset + 1], dq[offset + 2]);
            }
            return direction;
        }

        double appendContactGradientAndEnergy(const DERSubsystem& subsystem,
                                              const runtime::ContactStencil& contact,
                                              runtime::DofView* g)
        {
            const std::optional<LocalContactStencil> local =
                gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return 0.0;
            }

            const engine::contact::GIPCBarrierStencil barrier =
                makeBarrierStencil(*local, contact.type);
            if (g == nullptr)
            {
                return engine::contact::computeGIPCBarrierEnergy(barrier);
            }
            const engine::contact::LocalBarrierGradient barrier_gradient =
                engine::contact::computeGIPCBarrierGradient(barrier);
            scatterBarrierValues(*local, barrier_gradient.gradient, *g);
            return barrier_gradient.energy;
        }

        void applyContactHessianProduct(const DERSubsystem& subsystem,
                                        const runtime::ContactStencil& contact,
                                        runtime::ConstDofView dq,
                                        runtime::DofView y)
        {
            const std::optional<LocalContactStencil> local =
                gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return;
            }

            const std::array<glm::dvec3, 4> product =
                engine::contact::computeGIPCBarrierHessianProduct(
                    makeBarrierStencil(*local, contact.type),
                    gatherBarrierDirection(*local, dq));
            scatterBarrierValues(*local, product, y);
        }

        void appendContactHessianTriplets(
            const DERSubsystem& subsystem,
            const runtime::ContactStencil& contact,
            std::vector<Eigen::Triplet<double>>& triplets)
        {
            const std::optional<LocalContactStencil> local =
                gatherLocalContact(subsystem, contact);
            if (!local)
            {
                return;
            }

            const engine::contact::LocalBarrierHessian hessian =
                engine::contact::computeGIPCBarrierHessian(
                    makeBarrierStencil(*local, contact.type));
            for (int row_point = 0; row_point < local->count; row_point++)
            {
                const int row_offset = local->qOffsets[row_point];
                for (int col_point = 0; col_point < local->count; col_point++)
                {
                    const int col_offset = local->qOffsets[col_point];
                    const glm::dmat3& block =
                        hessian.blocks[row_point][col_point];
                    for (int row = 0; row < 3; row++)
                    {
                        for (int col = 0; col < 3; col++)
                        {
                            const double value = block[col][row];
                            if (value != 0.0)
                            {
                                if (!std::isfinite(value))
                                {
                                    throw std::runtime_error(
                                        "DER internal contact Hessian contains a non-finite value");
                                }
                                triplets.emplace_back(row_offset + row,
                                                      col_offset + col,
                                                      value);
                            }
                        }
                    }
                }
            }
        }

        void dumpDenseMatrix(const Eigen::SparseMatrix<double>& matrix,
                             const char* path)
        {
            std::ofstream out(path);
            if (!out)
            {
                return;
            }

            out << "# DER CPU backend local matrix\n";
            out << "# format: dense rows, whitespace-separated values\n";
            out << "rows " << matrix.rows() << '\n';
            out << "cols " << matrix.cols() << '\n';
            out << "nonzeros " << matrix.nonZeros() << '\n';
            out << std::scientific << std::setprecision(17);
            const Eigen::MatrixXd dense = Eigen::MatrixXd(matrix);
            for (int row = 0; row < dense.rows(); ++row)
            {
                for (int col = 0; col < dense.cols(); ++col)
                {
                    if (col > 0)
                    {
                        out << ' ';
                    }
                    out << dense(row, col);
                }
                out << '\n';
            }
        }
    } // namespace

    DERCPUBackend::DERCPUBackend(DERSubsystem& subsystem)
        : subsystem_(subsystem)
    {
    }

    void DERCPUBackend::writeState(runtime::DofView q,
                                   runtime::DofView qdot) const
    {
        requireCPU(q, "DERCpuBackend::writeState");
        requireCPU(qdot, "DERCpuBackend::writeState");
        requireDofCount(q, subsystem_.localScalarCount(), "DERCpuBackend::writeState");
        requireDofCount(qdot, subsystem_.localScalarCount(), "DERCpuBackend::writeState");

        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            const int base = offsets[rod_index].q;
            for (int vertex = 0; vertex < static_cast<int>(rod.state().size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    q[base + 4 * vertex + lane] =
                        rod.state().blocks[vertex][lane];
                    qdot[base + 4 * vertex + lane] =
                        rod.velocity().blocks[vertex][lane];
                }
            }
        }
    }

    void DERCPUBackend::readState(runtime::ConstDofView q,
                                  runtime::ConstDofView qdot)
    {
        requireCPU(q, "DERCpuBackend::readState");
        requireCPU(qdot, "DERCpuBackend::readState");
        requireDofCount(q, subsystem_.localScalarCount(), "DERCpuBackend::readState");
        requireDofCount(qdot, subsystem_.localScalarCount(), "DERCpuBackend::readState");

        const auto& offsets = subsystem_.rodOffsets();
        auto& rods = subsystem_.rods();
        for (int rod_index = 0; rod_index < rods.size(); ++rod_index)
        {
            Rod& rod = rods[rod_index];
            const int base = offsets[rod_index].q;
            for (int vertex = 0; vertex < rod.state().size(); ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    rod.state().blocks[vertex][lane] =
                        q[base + 4 * vertex + lane];
                    rod.velocity().blocks[vertex][lane] =
                        qdot[base + 4 * vertex + lane];
                }
            }
        }
    }

    void DERCPUBackend::beginStep(runtime::ConstDofView q,
                                  runtime::ConstDofView qdot,
                                  double dt)
    {
        requireCPU(q, "DERCpuBackend::beginStep");
        requireCPU(qdot, "DERCpuBackend::beginStep");
        step_dt_ = dt;
        step_start_ = gatherLocalVector(q);
        inertial_target_ = step_start_ + dt * gatherLocalVector(qdot);

        mass_diagonal_ = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        step_previous_states_.clear();
        step_previous_states_.reserve(rods.size());
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            step_previous_states_.emplace_back(rods[rod_index].state().blocks.begin(),
                                               rods[rod_index].state().blocks.end());
            const Eigen::VectorXd rod_mass = rods[rod_index].massDiagonal();
            const int base = offsets[rod_index].q;
            mass_diagonal_.segment(base, rod_mass.size()) = rod_mass;
        }
    }

    void DERCPUBackend::acceptStep(runtime::ConstDofView q,
                                   runtime::DofView qdot,
                                   double dt)
    {
        requireCPU(q, "DERCpuBackend::acceptStep");
        requireCPU(qdot, "DERCpuBackend::acceptStep");
        requireDofCount(q, subsystem_.localScalarCount(), "DERCpuBackend::acceptStep");
        requireDofCount(qdot, subsystem_.localScalarCount(), "DERCpuBackend::acceptStep");
        if (step_start_.size() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend step has not begun");
        }

        const int scalar_count = subsystem_.localScalarCount();
        const Eigen::VectorXd local_q = gatherLocalVector(q);
        const Eigen::VectorXd local_qdot = (local_q - step_start_) / dt;
        for (int i = 0; i < scalar_count; ++i)
        {
            qdot[i] = local_qdot[i];
        }

        auto& rods = subsystem_.rods();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            if (rod_index >= static_cast<int>(step_previous_states_.size()))
            {
                continue;
            }
            std::vector<RodDof>& previous_blocks =
                step_previous_states_[rod_index];
            if (previous_blocks.size() != rods[rod_index].state().size())
            {
                continue;
            }
            RodState previous{
                std::span<RodDof>(previous_blocks.data(), previous_blocks.size())
            };
            rods[rod_index].transportReferenceFrames(previous);
        }
    }

    double DERCPUBackend::evaluateObjective(runtime::ConstDofView q,
                                            runtime::ConstDofView qdot,
                                            double dt)
    {
        requireCPU(q, "DERCpuBackend::evaluateObjective");
        requireCPU(qdot, "DERCpuBackend::evaluateObjective");

        const Eigen::VectorXd local_q = gatherLocalVector(q);
        double objective = 0.0;
        const Eigen::VectorXd inertial_residual = local_q - inertial_target_;
        objective +=
            0.5 *
            (mass_diagonal_.array() * inertial_residual.array().square()).sum() /
            (dt * dt);

        const auto& rods = subsystem_.rods();
        for (int i = 0; i < rods.size(); i++)
        {
            const RodEvaluation evaluation = evaluateRod(rods[i], i);
            if (!evaluation.valid)
            {
                throw std::runtime_error(evaluation.diagnostic);
            }
            objective += evaluation.energy.total();
        }
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            objective += appendContactGradientAndEnergy(subsystem_, contact, nullptr);
        }
        return objective;
    }

    void DERCPUBackend::updateInternalConstraints(double time, double dt)
    {
    }

    void DERCPUBackend::prepareLocalOperator(double dt)
    {
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        evaluations_.clear();
        evaluations_.reserve(rods.size());

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(
            subsystem_.localScalarCount() +
            144 * subsystem_.internal_contacts_.size()));
        if (mass_diagonal_.size() != subsystem_.localScalarCount())
        {
            mass_diagonal_ = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
            for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
                 ++rod_index)
            {
                const Eigen::VectorXd rod_mass = rods[rod_index].massDiagonal();
                const int base = offsets[rod_index].q;
                mass_diagonal_.segment(base, rod_mass.size()) = rod_mass;
            }
        }
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            RodEvaluation evaluation = evaluateRod(rod, rod_index);
            if (!evaluation.valid)
            {
                throw std::runtime_error(evaluation.diagnostic);
            }

            const int base = offsets[rod_index].q;
            const Eigen::VectorXd mass = rod.massDiagonal();
            Eigen::MatrixXd mass_matrix =
                Eigen::MatrixXd::Zero(mass.size(), mass.size());
            for (Eigen::Index i = 0; i < mass.size(); ++i)
            {
                mass_matrix(i, i) = mass[i] / (dt * dt);
            }
            const Eigen::MatrixXd rod_hessian =
                Eigen::MatrixXd(evaluation.hessian);
            diagnoseDenseAssemblyMatrix(
                "rod_mass", mass_matrix, subsystem_.id(), rod_index, true);
            diagnoseDenseAssemblyMatrix(
                "rod_hessian", rod_hessian, subsystem_.id(), rod_index, false);
            diagnoseDenseAssemblyMatrix(
                "rod_mass_plus_hessian",
                mass_matrix + rod_hessian,
                subsystem_.id(),
                rod_index,
                true);
            for (Eigen::Index i = 0; i < mass.size(); ++i)
            {
                triplets.emplace_back(base + static_cast<int>(i),
                                      base + static_cast<int>(i),
                                      mass[i] / (dt * dt));
            }
            for (int outer = 0; outer < evaluation.hessian.outerSize(); ++outer)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(evaluation.hessian,
                                                                   outer);
                     it; ++it)
                {
                    triplets.emplace_back(base + static_cast<int>(it.row()),
                                          base + static_cast<int>(it.col()),
                                          it.value());
                }
            }

            evaluations_.push_back(std::move(evaluation));
        }
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            appendContactHessianTriplets(subsystem_, contact, triplets);
        }

        local_solver_ready_ = false;
        local_matrix_.resize(subsystem_.localScalarCount(),
                             subsystem_.localScalarCount());
        local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        local_matrix_.makeCompressed();
        diagnoseLocalMatrixSPD(
            local_matrix_,
            subsystem_.id(),
            static_cast<int>(subsystem_.internal_contacts_.size()));
        local_solver_.compute(local_matrix_);
        if (local_solver_.info() != Eigen::Success)
        {
            dumpDenseMatrix(local_matrix_, "der-local-matrix-factor-failure.txt");
            throw std::runtime_error("failed to factor DER CPU backend local matrix");
        }
        local_solver_ready_ = true;
    }

    void DERCPUBackend::assembleLocalGradient(runtime::DofView g) const
    {
        if (evaluations_.size() != subsystem_.rods().size())
        {
            throw std::runtime_error("DER CPU backend local operator is not prepared");
        }
        requireCPU(g, "DERCpuBackend::assembleLocalGradient");

        const auto& offsets = subsystem_.rodOffsets();
        const int scalar_count = subsystem_.localScalarCount();
        requireDofCount(g, scalar_count, "DERCpuBackend::assembleLocalGradient");

        Eigen::VectorXd local_gradient = Eigen::VectorXd::Zero(scalar_count);
        if (inertial_target_.size() == scalar_count &&
            mass_diagonal_.size() == scalar_count)
        {
            local_gradient +=
                (mass_diagonal_.array() *
                    (gatherCurrentState() - inertial_target_).array()).matrix() /
                (step_dt_ * step_dt_);
        }

        for (int rod_index = 0; rod_index < static_cast<int>(subsystem_.rods().size());
             ++rod_index)
        {
            const int base = offsets[rod_index].q;
            const RodEvaluation& evaluation = evaluations_[rod_index];
            for (int vertex = 0; vertex < static_cast<int>(evaluation.gradient.size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    local_gradient[base + 4 * vertex + lane] +=
                        evaluation.gradient[vertex][lane];
                }
            }
        }
        for (int i = 0; i < scalar_count; ++i)
        {
            addToVector(g, i, local_gradient[i]);
        }
        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            appendContactGradientAndEnergy(subsystem_, contact, &g);
        }
    }

    void DERCPUBackend::applyLocalMatrix(runtime::ConstDofView x,
                                         runtime::DofView y) const
    {
        if (local_matrix_.rows() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend local matrix is not prepared");
        }
        requireCPU(x, "DERCpuBackend::applyLocalMatrix");
        requireCPU(y, "DERCpuBackend::applyLocalMatrix");
        requireDofCount(x, subsystem_.localScalarCount(), "DERCpuBackend::applyLocalMatrix");
        requireDofCount(y, subsystem_.localScalarCount(), "DERCpuBackend::applyLocalMatrix");

        const int scalar_count = subsystem_.localScalarCount();
        const Eigen::VectorXd local_x = gatherLocalVector(x);
        const Eigen::VectorXd local_y = local_matrix_ * local_x;
        for (int i = 0; i < scalar_count; ++i)
        {
            addToVector(y, i, local_y[i]);
        }
    }

    void DERCPUBackend::solveLocalSystem(runtime::ConstDofView b,
                                         runtime::DofView x) const
    {
        if (local_matrix_.rows() != subsystem_.localScalarCount())
        {
            throw std::runtime_error("DER CPU backend local matrix is not prepared");
        }
        if (!local_solver_ready_)
        {
            throw std::runtime_error("DER CPU backend local solver is not prepared");
        }
        requireCPU(b, "DERCpuBackend::solveLocalSystem");
        requireCPU(x, "DERCpuBackend::solveLocalSystem");
        requireDofCount(b, subsystem_.localScalarCount(), "DERCpuBackend::solveLocalSystem");
        requireDofCount(x, subsystem_.localScalarCount(), "DERCpuBackend::solveLocalSystem");

        const Eigen::VectorXd local_b = gatherLocalVector(b);
        if (!local_b.allFinite())
        {
            spdlog::warn(
                "DER local solve received non-finite rhs: subsystem={} "
                "scalars={} rhs_norm={} rhs_inf={} internal_contacts={}",
                subsystem_.id(),
                subsystem_.localScalarCount(),
                local_b.norm(),
                vectorInfNorm(local_b),
                subsystem_.internal_contacts_.size());
            dumpDenseVector(local_b, "der-local-solve-rhs-nonfinite.txt");
        }

        const Eigen::VectorXd local_x = local_solver_.solve(local_b);
        if (local_solver_.info() != Eigen::Success)
        {
            dumpDenseMatrix(local_matrix_, "der-local-matrix-solve-failure.txt");
            dumpDenseVector(local_b, "der-local-solve-rhs-failure.txt");
            throw std::runtime_error("failed to solve DER CPU backend local system");
        }
        const Eigen::VectorXd local_ax = local_matrix_ * local_x;
        const Eigen::VectorXd solve_residual = local_ax - local_b;
        const double preconditioner_dot = local_b.dot(local_x);
        const double quadratic = local_x.dot(local_ax);
        const double rhs_norm = local_b.norm();
        const double solution_norm = local_x.norm();
        const double relative_solve_residual =
            solve_residual.norm() / std::max(1.0, rhs_norm);
        if (!local_x.allFinite() ||
            !std::isfinite(preconditioner_dot) ||
            preconditioner_dot <= 0.0 ||
            !std::isfinite(quadratic) ||
            quadratic <= 0.0 ||
            !std::isfinite(relative_solve_residual) ||
            relative_solve_residual > 1.0e-6)
        {
            const int max_rhs_scalar = maxAbsIndex(local_b);
            const int max_x_scalar = maxAbsIndex(local_x);
            const ScalarLocation max_rhs_location =
                locateScalar(subsystem_, max_rhs_scalar);
            const ScalarLocation max_x_location =
                locateScalar(subsystem_, max_x_scalar);
            spdlog::warn(
                "DER local solve diagnostic: subsystem={} scalars={} "
                "b_dot_x={} xTAx={} rhs_norm={} x_norm={} rhs_inf={} "
                "x_inf={} relative_residual={} internal_contacts={} "
                "max_rhs_scalar={} max_rhs_rod={} max_rhs_vertex={} "
                "max_rhs_lane={} max_x_scalar={} max_x_rod={} "
                "max_x_vertex={} max_x_lane={}",
                subsystem_.id(),
                subsystem_.localScalarCount(),
                preconditioner_dot,
                quadratic,
                rhs_norm,
                solution_norm,
                vectorInfNorm(local_b),
                vectorInfNorm(local_x),
                relative_solve_residual,
                subsystem_.internal_contacts_.size(),
                max_rhs_scalar,
                max_rhs_location.rod,
                max_rhs_location.vertex,
                max_rhs_location.lane,
                max_x_scalar,
                max_x_location.rod,
                max_x_location.vertex,
                max_x_location.lane);
            dumpDenseMatrix(local_matrix_, "der-local-matrix-invalid-solve.txt");
            dumpDenseVector(local_b, "der-local-solve-rhs-invalid.txt");
            dumpDenseVector(local_x, "der-local-solve-x-invalid.txt");
        }

        const int scalar_count = subsystem_.localScalarCount();
        for (int i = 0; i < scalar_count; ++i)
        {
            x[i] = local_x[i];
        }
    }

    void DERCPUBackend::mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                               runtime::GeometryView globalDx) const
    {
        requireCPU(localDq, "DERCpuBackend::mapLocalDirectionToGeometry");
        requireCPU(globalDx, "DERCpuBackend::mapLocalDirectionToGeometry");

        const auto& samples = subsystem_.geometrySamples();
        const auto& geometry_points = subsystem_.geometryPointIds();
        if (geometry_points.size() != samples.size())
        {
            throw std::runtime_error("DER geometry point mapping is stale");
        }

        for (int sample = 0; sample < samples.size(); sample++)
        {
            auto point = geometry_points[sample];
            if (point < 0 || point >= globalDx.pointCount())
            {
                throw std::invalid_argument("DER global geometry direction buffer is too small");
            }
            const int offset = samples[sample].qOffset;
            globalDx[point] = glm::dvec3(localDq[offset + 0],
                                         localDq[offset + 1],
                                         localDq[offset + 2]);
        }
    }

    void DERCPUBackend::scatterContactGradient(
        std::span<const runtime::PointIdx> points,
        runtime::ConstGeometryView pointGradient,
        runtime::DofView g) const
    {
        requireCPU(pointGradient, "DERCpuBackend::scatterContactGradient");
        requireCPU(g, "DERCpuBackend::scatterContactGradient");

        if (points.size() != static_cast<size_t>(pointGradient.pointCount()))
        {
            throw std::invalid_argument("contact point and gradient counts differ");
        }

        requireDofCount(g, subsystem_.localScalarCount(), "DERCpuBackend::scatterContactGradient");

        const auto& geometry_points = subsystem_.geometryPointIds();
        const auto& samples = subsystem_.geometrySamples();
        for (size_t i = 0; i < points.size(); ++i)
        {
            const auto it =
                std::find(geometry_points.begin(), geometry_points.end(), points[i]);
            if (it == geometry_points.end())
            {
                continue;
            }
            const int sample = static_cast<int>(it - geometry_points.begin());
            const int offset = samples[sample].qOffset;
            const glm::dvec3 point_gradient =
                pointGradient[static_cast<int>(i)];
            addToVector(g, offset + 0, point_gradient.x);
            addToVector(g, offset + 1, point_gradient.y);
            addToVector(g, offset + 2, point_gradient.z);
        }
    }

    void DERCPUBackend::applyContactGeometryHessianProduct(
        std::span<const runtime::PointIdx> productPoints,
        runtime::ConstGeometryView pointHessianProduct,
        runtime::DofView localY) const
    {
        requireCPU(pointHessianProduct, "DERCpuBackend::applyContactGeometryHessianProduct");
        requireCPU(localY, "DERCpuBackend::applyContactGeometryHessianProduct");

        scatterContactGradient(productPoints, pointHessianProduct, localY);
    }

    void DERCPUBackend::applyInternalContactHessian(
        runtime::ConstDofView localDq,
        runtime::DofView localY) const
    {
        requireCPU(localDq, "DERCpuBackend::applyInternalContactHessian");
        requireCPU(localY, "DERCpuBackend::applyInternalContactHessian");
        requireDofCount(localDq, subsystem_.localScalarCount(),
                        "DERCpuBackend::applyInternalContactHessian");
        requireDofCount(localY, subsystem_.localScalarCount(),
                        "DERCpuBackend::applyInternalContactHessian");

        for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_)
        {
            applyContactHessianProduct(subsystem_, contact, localDq, localY);
        }
    }

    const RodEvaluation& DERCPUBackend::cachedEvaluation(int rod) const
    {
        return evaluations_.at(static_cast<size_t>(rod));
    }

    Eigen::VectorXd DERCPUBackend::gatherLocalVector(
        runtime::ConstDofView values) const
    {
        requireDofCount(values, subsystem_.localScalarCount(),
                        "DERCPUBackend::gatherLocalVector");
        Eigen::VectorXd local(subsystem_.localScalarCount());
        for (int i = 0; i < subsystem_.localScalarCount(); ++i)
        {
            local[i] = values[i];
        }
        return local;
    }

    Eigen::VectorXd DERCPUBackend::gatherCurrentState() const
    {
        Eigen::VectorXd local = Eigen::VectorXd::Zero(subsystem_.localScalarCount());
        const auto& rods = subsystem_.rods();
        const auto& offsets = subsystem_.rodOffsets();
        for (int rod_index = 0; rod_index < static_cast<int>(rods.size());
             ++rod_index)
        {
            const Rod& rod = rods[rod_index];
            const int base = offsets[rod_index].q;
            for (int vertex = 0; vertex < static_cast<int>(rod.state().size());
                 ++vertex)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    local[base + 4 * vertex + lane] = rod.state().blocks[vertex][lane];
                }
            }
        }
        return local;
    }

    RodEvaluation DERCPUBackend::evaluateRod(const Rod& rod, int rodIndex)
    {
        if (rodIndex >= 0 &&
            rodIndex < static_cast<int>(step_previous_states_.size()))
        {
            std::vector<RodDof>& previous_blocks =
                step_previous_states_[rodIndex];
            if (previous_blocks.size() == rod.state().size())
            {
                RodState previous{
                    std::span<RodDof>(previous_blocks.data(), previous_blocks.size())
                };
                return rod.evaluate(subsystem_.gravity(), previous);
            }
        }
        return rod.evaluate(subsystem_.gravity());
    }

    void DERCPUBackend::addToVector(runtime::DofView values,
                                    int localOffset,
                                    double value) const
    {
        values[localOffset] += value;
    }
} // namespace ksk::der
