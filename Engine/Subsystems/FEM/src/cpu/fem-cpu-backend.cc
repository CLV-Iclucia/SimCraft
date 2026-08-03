#include <Contact/contact-barrier.h>
#include <Core/profiler.h>
#include <Runtime/contact-barrier-energy.h>
#include <FEM/cpu/fem-cpu-backend.h>
#include <FEM/fem-subsystem.h>
#include <tbb/parallel_for.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/geometric.hpp>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ksk::engine::fem {
namespace {
void requireCPU(runtime::ConstDofView view, const char* operation) {
#ifndef NDEBUG
  if (!view.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU DofView");
  }
#endif
}

void requireCPU(runtime::DofView view, const char* operation) {
  requireCPU(view.asConst(), operation);
}

void requireCPU(runtime::ConstGeometryView view, const char* operation) {
#ifndef NDEBUG
  if (!view.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU GeometryView");
  }
#endif
}

void requireCPU(runtime::GeometryView view, const char* operation) {
  requireCPU(view.asConst(), operation);
}

void requireDofCount(runtime::ConstDofView view, int expected,
                     const char* operation) {
  if (view.scalarCount() < expected) {
    throw std::invalid_argument(std::string(operation) +
                                " received a DOF view that is too small");
  }
}

Eigen::VectorXd gatherLocalVector(runtime::ConstDofView view, int scalarCount) {
  requireDofCount(view, scalarCount, "gatherLocalVector");
  Eigen::VectorXd local(scalarCount);
  for (int i = 0; i < scalarCount; ++i) {
    local[i] = view[i];
  }
  return local;
}

int contactPointCount(runtime::ContactCase type) {
  switch (type) {
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

const char* contactCaseName(runtime::ContactCase type) {
  switch (type) {
    case runtime::ContactCase::PP:
      return "PP";
    case runtime::ContactCase::PE:
      return "PE";
    case runtime::ContactCase::PT:
      return "PT";
    case runtime::ContactCase::EE:
      return "EE";
  }
  return "unknown";
}

double dofViewNorm(runtime::ConstDofView view) {
  double squared_norm = 0.0;
  for (const double value : view.cpu()) {
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

contact::EBarrierStencil barrierStencilType(runtime::ContactCase type) {
  switch (type) {
    case runtime::ContactCase::PP:
      return contact::EBarrierStencil::PP;
    case runtime::ContactCase::PE:
      return contact::EBarrierStencil::PE;
    case runtime::ContactCase::PT:
      return contact::EBarrierStencil::PT;
    case runtime::ContactCase::EE:
      return contact::EBarrierStencil::EE;
  }
  return contact::EBarrierStencil::PP;
}

int findSampleIndex(const FEMSubsystem& subsystem, runtime::PointIdx point) {
  const std::vector<runtime::PointIdx>& points = subsystem.geometryPointIds();
  const auto it = std::find(points.begin(), points.end(), point);
  if (it == points.end()) {
    return -1;
  }
  return static_cast<int>(it - points.begin());
}

glm::dvec3 localVertexPosition(const FEMSubsystem& subsystem,
                               const FEMVertexSample& sample) {
  return subsystem.vertexPosition(sample.mesh, sample.vertex);
}

glm::dvec3 localRestVertexPosition(const FEMSubsystem& subsystem,
                                   const FEMVertexSample& sample) {
  return subsystem.restVertexPosition(sample.mesh, sample.vertex);
}

double triangleArea(const std::vector<glm::dvec3>& vertices,
                    const std::array<int, 3>& triangle) {
  const glm::dvec3 e0 = vertices.at(static_cast<size_t>(triangle[1])) -
                        vertices.at(static_cast<size_t>(triangle[0]));
  const glm::dvec3 e1 = vertices.at(static_cast<size_t>(triangle[2])) -
                        vertices.at(static_cast<size_t>(triangle[0]));
  return 0.5 * glm::length(glm::cross(e0, e1));
}

struct LocalContactStencil {
  std::array<int, 4> qOffsets{-1, -1, -1, -1};
  std::array<glm::dvec3, 4> x{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                              glm::dvec3(0.0)};
  std::array<glm::dvec3, 4> restX{glm::dvec3(0.0), glm::dvec3(0.0),
                                  glm::dvec3(0.0), glm::dvec3(0.0)};
  int count = 0;
  double dHat = 0.0;
  double surfaceOffset = 0.0;
  double kappa = 0.0;
};

std::optional<LocalContactStencil> gatherLocalContact(
    const FEMSubsystem& subsystem, const runtime::ContactStencil& contact) {
  LocalContactStencil local;
  local.count = contactPointCount(contact.type);
  local.dHat = contact.dHat;
  local.surfaceOffset = contact.thickness;
  local.kappa = contact.stiffness;

  if (local.count <= 0 || local.dHat <= 0.0 || local.kappa <= 0.0) {
    return std::nullopt;
  }

  for (int i = 0; i < local.count; ++i) {
    const int sample_index =
        findSampleIndex(subsystem, contact.geometryIds[static_cast<size_t>(i)]);
    if (sample_index < 0) {
      return std::nullopt;
    }
    const FEMVertexSample& sample = subsystem.samples().at(sample_index);
    local.qOffsets[i] = sample.qOffset;
    local.x[i] = localVertexPosition(subsystem, sample);
    local.restX[i] = localRestVertexPosition(subsystem, sample);
  }
  return local;
}

ksk::engine::contact::GIPCBarrierStencil makeBarrierStencil(
    const LocalContactStencil& contact, runtime::ContactCase type) {
  ksk::engine::contact::GIPCBarrierStencil barrier;
  barrier.type = barrierStencilType(type);
  barrier.x = contact.x;
  barrier.restX = contact.restX;
  barrier.dHat = contact.dHat;
  barrier.reservedDist = contact.surfaceOffset;
  barrier.kappa = contact.kappa;
  return barrier;
}

void scatterBarrierValues(const LocalContactStencil& contact,
                          const std::array<glm::dvec3, 4>& values,
                          runtime::DofView y) {
  for (int i = 0; i < contact.count; i++) {
    const int offset = contact.qOffsets[i];
    y[offset + 0] += values[i].x;
    y[offset + 1] += values[i].y;
    y[offset + 2] += values[i].z;
  }
}

std::array<glm::dvec3, 4> gatherBarrierDirection(
    const LocalContactStencil& contact, runtime::ConstDofView dq) {
  std::array direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  for (int i = 0; i < contact.count; i++) {
    const int offset = contact.qOffsets[i];
    direction[i] = glm::dvec3(dq[offset + 0], dq[offset + 1], dq[offset + 2]);
  }
  return direction;
}

double appendContactGradientAndEnergy(const FEMSubsystem& subsystem,
                                      const runtime::ContactStencil& contact,
                                      runtime::DofView* g) {
  const std::optional<LocalContactStencil> local =
      gatherLocalContact(subsystem, contact);
  if (!local) return 0.0;

  const contact::GIPCBarrierStencil barrier =
      makeBarrierStencil(*local, contact.type);

  if (g == nullptr) {
    return ksk::engine::contact::computeGIPCBarrierEnergy(barrier) +
           runtime::computeContactFrictionEnergyGradient(
               contact, std::span(local->x.data(), local->count), nullptr);
  }

  const contact::LocalBarrierGradient barrier_gradient =
      ksk::engine::contact::computeGIPCBarrierGradient(barrier);
  std::array<glm::dvec3, 4> total_gradient = barrier_gradient.gradient;
  const double friction_energy =
      runtime::computeContactFrictionEnergyGradient(
          contact, std::span(local->x.data(), local->count), &total_gradient);
  scatterBarrierValues(*local, total_gradient, *g);
  return barrier_gradient.energy + friction_energy;
}

void applyContactHessianProduct(const FEMSubsystem& subsystem,
                                const runtime::ContactStencil& contact,
                                runtime::ConstDofView dq, runtime::DofView y) {
  const std::optional<LocalContactStencil> local =
      gatherLocalContact(subsystem, contact);
  if (!local) {
    return;
  }

  const std::array<glm::dvec3, 4> product =
      ksk::engine::contact::computeGIPCBarrierHessianProduct(
          makeBarrierStencil(*local, contact.type),
          gatherBarrierDirection(*local, dq));
  std::array<glm::dvec3, 4> direction = gatherBarrierDirection(*local, dq);
  std::array<glm::dvec3, 4> total_product = product;
  const std::array<glm::dvec3, 4> friction_product =
      runtime::computeContactFrictionHessianProduct(
          contact, std::span(local->x.data(), local->count),
          std::span(direction.data(), local->count));
  for (int i = 0; i < local->count; ++i) {
    total_product[static_cast<size_t>(i)] +=
        friction_product[static_cast<size_t>(i)];
  }
  scatterBarrierValues(*local, total_product, y);
}

size_t contactHessianTripletCount(const FEMSubsystem& subsystem,
                                  const runtime::ContactStencil& contact) {
  const std::optional<LocalContactStencil> local =
      gatherLocalContact(subsystem, contact);
  if (!local) return 0;
  return static_cast<size_t>(local->count) * static_cast<size_t>(local->count) *
         9;
}

void fillContactHessianTriplets(const FEMSubsystem& subsystem,
                                const runtime::ContactStencil& contact,
                                std::span<Eigen::Triplet<double>> triplets) {
  const std::optional<LocalContactStencil> local =
      gatherLocalContact(subsystem, contact);
  if (!local) {
    assert(triplets.empty());
    return;
  }

  const ksk::engine::contact::LocalBarrierHessian hessian =
      ksk::engine::contact::computeGIPCBarrierHessian(
          makeBarrierStencil(*local, contact.type));
  const runtime::ContactLocalHessian friction_hessian =
      runtime::computeContactFrictionHessian(
          contact, std::span(local->x.data(), local->count));
  size_t index = 0;
  for (int row_point = 0; row_point < local->count; row_point++) {
    const int row_offset = local->qOffsets[row_point];
    for (int col_point = 0; col_point < local->count; col_point++) {
      const int col_offset = local->qOffsets[col_point];
      const glm::dmat3 block =
          hessian.blocks[row_point][col_point] +
          friction_hessian.blocks[row_point][col_point];
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          triplets[index++] = Eigen::Triplet<double>(
              row_offset + row, col_offset + col, block[col][row]);
        }
      }
    }
  }
  assert(index == triplets.size());
}

double gravityPotential(const Eigen::VectorXd& localQ,
                        const Eigen::VectorXd& massDiagonal,
                        const glm::dvec3& gravity) {
  double energy = 0.0;
  for (int vertex = 0; 3 * vertex + 2 < localQ.size(); ++vertex) {
    const int offset = 3 * vertex;
    const double mass = massDiagonal[offset];
    energy -= mass *
              (gravity.x * localQ[offset + 0] + gravity.y * localQ[offset + 1] +
               gravity.z * localQ[offset + 2]);
  }
  return energy;
}

void addGravityGradient(Eigen::VectorXd& gradient,
                        const Eigen::VectorXd& massDiagonal,
                        const glm::dvec3& gravity) {
  for (int vertex = 0; 3 * vertex + 2 < gradient.size(); ++vertex) {
    const int offset = 3 * vertex;
    const double mass = massDiagonal[offset];
    gradient[offset + 0] -= mass * gravity.x;
    gradient[offset + 1] -= mass * gravity.y;
    gradient[offset + 2] -= mass * gravity.z;
  }
}
}  // namespace

FEMCPUBackend::FEMCPUBackend(FEMSubsystem& subsystem)
    : subsystem_(subsystem), mass_diagonal_(buildMassDiagonal()) {}

void FEMCPUBackend::writeState(runtime::DofView q,
                               runtime::DofView qdot) const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/WriteState");
  requireCPU(q, "FEMCPUBackend::writeState");
  requireCPU(qdot, "FEMCPUBackend::writeState");
  requireDofCount(q, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::writeState");
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::writeState");

  for (const FEMVertexSample& sample : subsystem_.samples()) {
    const glm::dvec3 position =
        subsystem_.vertexPosition(sample.mesh, sample.vertex);
    const glm::dvec3 velocity =
        subsystem_.vertexVelocity(sample.mesh, sample.vertex);
    q[sample.qOffset + 0] = position.x;
    q[sample.qOffset + 1] = position.y;
    q[sample.qOffset + 2] = position.z;
    qdot[sample.qOffset + 0] = velocity.x;
    qdot[sample.qOffset + 1] = velocity.y;
    qdot[sample.qOffset + 2] = velocity.z;
  }
}

void FEMCPUBackend::readState(runtime::ConstDofView q,
                              runtime::ConstDofView qdot) {
  KSK_PROFILE_SCOPE("FEMCPUBackend/ReadState");
  requireCPU(q, "FEMCPUBackend::readState");
  requireCPU(qdot, "FEMCPUBackend::readState");
  requireDofCount(q, subsystem_.range_.scalarCount, "FEMCPUBackend::readState");
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::readState");

  for (const FEMVertexSample& sample : subsystem_.samples()) {
    subsystem_.setVertexPosition(
        sample.mesh, sample.vertex,
        glm::dvec3(q[sample.qOffset + 0], q[sample.qOffset + 1],
                   q[sample.qOffset + 2]));
    subsystem_.setVertexVelocity(
        sample.mesh, sample.vertex,
        glm::dvec3(qdot[sample.qOffset + 0], qdot[sample.qOffset + 1],
                   qdot[sample.qOffset + 2]));
  }
  subsystem_.rebuildCompatibilityViews();
}

void FEMCPUBackend::beginStep(runtime::ConstDofView q,
                              runtime::ConstDofView qdot, double dt) {
  KSK_PROFILE_SCOPE("FEMCPUBackend/BeginStep");
  step_dt_ = dt;
  step_start_ = gatherLocalVector(q, subsystem_.range_.scalarCount);
  inertial_target_ =
      step_start_ + dt * gatherLocalVector(qdot, subsystem_.range_.scalarCount);
  mass_diagonal_ = buildMassDiagonal();
}

void FEMCPUBackend::acceptStep(runtime::ConstDofView q, runtime::DofView qdot,
                               double dt) {
  KSK_PROFILE_SCOPE("FEMCPUBackend/AcceptStep");
  if (step_start_.size() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend step has not begun");
  }

  const Eigen::VectorXd local_q =
      gatherLocalVector(q, subsystem_.range_.scalarCount);
  const Eigen::VectorXd local_qdot = (local_q - step_start_) / dt;
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::acceptStep");
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    qdot[i] = local_qdot[i];
  }
}

double FEMCPUBackend::evaluateObjective(runtime::ConstDofView q,
                                        runtime::ConstDofView qdot, double dt) {
  SIM_PROFILE_SCOPE_COLOR("FEMCPUBackend/EvaluateObjective",
                          ksk::core::profiler_colors::kPurple);
  (void)qdot;
  const Eigen::VectorXd local_q =
      gatherLocalVector(q, subsystem_.range_.scalarCount);
  const Eigen::VectorXd residual = local_q - inertial_target_;
  double objective =
      0.5 * (mass_diagonal_.array() * residual.array().square()).sum() /
      (dt * dt);
  objective += subsystem_.elasticEnergy(local_q);
  objective += gravityPotential(local_q, mass_diagonal_, subsystem_.gravity_);

  for (const FEMSubsystem::ActiveConstraint& constraint :
       subsystem_.active_constraints_) {
    const double value = local_q[constraint.qOffset];
    const double diff = value - constraint.target;
    objective += 0.5 * constraint.stiffness * diff * diff;
  }
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/EvaluateObjective/InternalContacts");
    for (const runtime::ContactStencil& contact :
         subsystem_.internal_contacts_) {
      objective += appendContactGradientAndEnergy(subsystem_, contact, nullptr);
    }
  }
  return objective;
}

void FEMCPUBackend::updateInternalConstraints(double time, double dt) {
  subsystem_.updateConstraintTargets(time);
}

void FEMCPUBackend::prepareLocalOperator(double dt) {
  SIM_PROFILE_SCOPE_COLOR("FEMCPUBackend/PrepareLocalOperator",
                          ksk::core::profiler_colors::kCyan);
  if (mass_diagonal_.size() != subsystem_.range_.scalarCount) {
    mass_diagonal_ = buildMassDiagonal();
  }

  const Eigen::VectorXd local_q = subsystem_.gatherCurrentState();
  std::vector<Eigen::Triplet<double>> triplets;

  const size_t mass_offset = 0;
  const size_t mass_count = static_cast<size_t>(subsystem_.range_.scalarCount);
  const size_t constraint_offset = mass_offset + mass_count;
  const size_t constraint_count = subsystem_.active_constraints_.size();
  const size_t elastic_offset = constraint_offset + constraint_count;
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/CountStatic");
    triplets.resize(elastic_offset);
  }
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/FillMass");
    const double inv_dt_sqr = 1.0 / (dt * dt);
    tbb::parallel_for(size_t{0}, mass_count, [&](size_t index) {
      const int row = static_cast<int>(index);
      triplets[mass_offset + index] =
          Eigen::Triplet<double>(row, row, mass_diagonal_[row] * inv_dt_sqr);
    });
  }
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/FillConstraints");
    tbb::parallel_for(size_t{0}, constraint_count, [&](size_t index) {
      const FEMSubsystem::ActiveConstraint& constraint =
          subsystem_.active_constraints_[index];
      triplets[constraint_offset + index] = Eigen::Triplet<double>(
          constraint.qOffset, constraint.qOffset, constraint.stiffness);
    });
  }
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/Elastic");
    assert(triplets.size() == elastic_offset);
    subsystem_.assembleElasticHessian(local_q, triplets);
  }
  const size_t internal_contact_offset = triplets.size();
  {
    KSK_PROFILE_SCOPE(
        "FEMCPUBackend/PrepareLocalOperator/InternalContacts/Count");
    std::vector<size_t> counts(subsystem_.internal_contacts_.size(), 0);
    tbb::parallel_for(size_t{0}, counts.size(), [&](size_t index) {
      counts[index] = contactHessianTripletCount(
          subsystem_, subsystem_.internal_contacts_[index]);
    });

    std::vector<size_t> offsets(counts.size(), 0);
    {
      KSK_PROFILE_SCOPE(
          "FEMCPUBackend/PrepareLocalOperator/InternalContacts/Scan");
      std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(),
                          internal_contact_offset);
    }
    const size_t contact_triplet_count =
        counts.empty() ? 0 : offsets.back() + counts.back() -
                                internal_contact_offset;
    triplets.resize(internal_contact_offset + contact_triplet_count);

    {
      KSK_PROFILE_SCOPE(
          "FEMCPUBackend/PrepareLocalOperator/InternalContacts/FillTriplets");
      tbb::parallel_for(size_t{0}, counts.size(), [&](size_t index) {
        if (counts[index] == 0) return;
        fillContactHessianTriplets(
            subsystem_, subsystem_.internal_contacts_[index],
            std::span<Eigen::Triplet<double>>(triplets.data() + offsets[index],
                                              counts[index]));
      });
    }
  }

  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/CompressMatrix");
    local_solver_ready_ = false;
    local_matrix_.resize(subsystem_.range_.scalarCount,
                         subsystem_.range_.scalarCount);
    local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
  }
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/PrepareLocalOperator/Factorize");
    local_solver_.compute(local_matrix_);
    if (local_solver_.info() != Eigen::Success) {
      throw std::runtime_error("failed to factor FEM CPU backend local matrix");
    }
    local_solver_ready_ = true;
  }
}

void FEMCPUBackend::assembleLocalGradient(runtime::DofView g) const {
  SIM_PROFILE_SCOPE_COLOR("FEMCPUBackend/AssembleLocalGradient",
                          ksk::core::profiler_colors::kRed);
  requireCPU(g, "FEMCPUBackend::assembleLocalGradient");
  requireDofCount(g, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::assembleLocalGradient");
  if (step_dt_ <= 0.0) {
    throw std::runtime_error("FEM CPU backend step has not begun");
  }

  const Eigen::VectorXd current = subsystem_.gatherCurrentState();
  Eigen::VectorXd local_gradient =
      (mass_diagonal_.array() * (current - inertial_target_).array()).matrix() /
      (step_dt_ * step_dt_);
  const double inertial_norm = local_gradient.norm();
  subsystem_.assembleElasticGradient(current, local_gradient);
  const double after_elastic_norm = local_gradient.norm();
  addGravityGradient(local_gradient, mass_diagonal_, subsystem_.gravity_);
  const double after_gravity_norm = local_gradient.norm();
  for (const FEMSubsystem::ActiveConstraint& constraint :
       subsystem_.active_constraints_) {
    local_gradient[constraint.qOffset] +=
        constraint.stiffness *
        (current[constraint.qOffset] - constraint.target);
  }
  const double after_constraints_norm = local_gradient.norm();
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    g[i] += local_gradient[i];
  }
  double max_internal_contact_delta = 0.0;
  const runtime::ContactStencil* worst_internal_contact = nullptr;
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/AssembleLocalGradient/InternalContacts");
    for (const runtime::ContactStencil& contact :
         subsystem_.internal_contacts_) {
      const double before_contact_norm = dofViewNorm(g.asConst());
      appendContactGradientAndEnergy(subsystem_, contact, &g);
      const double after_contact_norm = dofViewNorm(g.asConst());
      const double delta = std::abs(after_contact_norm - before_contact_norm);
      if (delta > max_internal_contact_delta) {
        max_internal_contact_delta = delta;
        worst_internal_contact = &contact;
      }
    }
  }
  const double final_norm = dofViewNorm(g.asConst());
  if (!std::isfinite(final_norm) || final_norm > 1.0e10) {
    std::cerr << "[fem-cpu] suspicious local gradient subsystem="
              << subsystem_.id_ << " scalars=" << subsystem_.range_.scalarCount
              << " inertial=" << inertial_norm
              << " after_elastic=" << after_elastic_norm
              << " after_gravity=" << after_gravity_norm
              << " after_constraints=" << after_constraints_norm
              << " final_with_internal_contacts=" << final_norm
              << " internal_contacts=" << subsystem_.internal_contacts_.size()
              << " active_constraints=" << subsystem_.active_constraints_.size()
              << '\n';
    if (worst_internal_contact != nullptr) {
      std::cerr << "[fem-cpu] worst internal contact type="
                << contactCaseName(worst_internal_contact->type) << " ids=[";
      const int point_count = contactPointCount(worst_internal_contact->type);
      for (int i = 0; i < point_count; ++i) {
        if (i > 0) {
          std::cerr << ',';
        }
        std::cerr
            << worst_internal_contact->geometryIds[static_cast<size_t>(i)];
      }
      std::cerr << "] delta_norm=" << max_internal_contact_delta
                << " dHat=" << worst_internal_contact->dHat
                << " thickness=" << worst_internal_contact->thickness
                << " stiffness=" << worst_internal_contact->stiffness << '\n';
    }
  }
}

void FEMCPUBackend::applyLocalMatrix(runtime::ConstDofView x,
                                     runtime::DofView y) const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/ApplyLocalMatrix");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend local operator is not prepared");
  }

  requireCPU(x, "FEMCPUBackend::applyLocalMatrix");
  requireCPU(y, "FEMCPUBackend::applyLocalMatrix");
  requireDofCount(x, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::applyLocalMatrix");
  requireDofCount(y, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::applyLocalMatrix");
  Eigen::Map<const Eigen::VectorXd> local_x(x.cpu().data(),
                                            subsystem_.range_.scalarCount);
  Eigen::Map<Eigen::VectorXd> local_y(y.cpu().data(),
                                      subsystem_.range_.scalarCount);
  local_y.noalias() += local_matrix_ * local_x;
}

void FEMCPUBackend::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/AppendLocalMatrixEntries");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("FEM CPU backend local operator is not prepared");
  }

  const int offset = subsystem_.range_.scalarOffset;
  entries.reserve(entries.size() +
                  static_cast<size_t>(local_matrix_.nonZeros()));
  for (int outer = 0; outer < local_matrix_.outerSize(); ++outer) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(local_matrix_, outer);
         it; ++it) {
      entries.push_back(runtime::SparseMatrixEntry{
          .row = offset + static_cast<int>(it.row()),
          .col = offset + static_cast<int>(it.col()),
          .value = it.value(),
      });
    }
  }
}

void FEMCPUBackend::solveLocalSystem(runtime::ConstDofView b,
                                     runtime::DofView x) const {
  SIM_PROFILE_SCOPE_COLOR("FEMCPUBackend/SolveLocalSystem",
                          ksk::core::profiler_colors::kBlue);
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/SolveLocalSystem/Validate");
    requireCPU(b, "FEMCPUBackend::solveLocalSystem");
    requireCPU(x, "FEMCPUBackend::solveLocalSystem");
    requireDofCount(b, subsystem_.range_.scalarCount,
                    "FEMCPUBackend::solveLocalSystem");
    requireDofCount(x, subsystem_.range_.scalarCount,
                    "FEMCPUBackend::solveLocalSystem");
    if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
      throw std::runtime_error(
          "FEM CPU backend local operator is not prepared");
    }
    if (!local_solver_ready_) {
      throw std::runtime_error("FEM CPU backend local solver is not prepared");
    }
  }

  Eigen::VectorXd local_b;
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/SolveLocalSystem/GatherRhs");
    local_b = gatherLocalVector(b, subsystem_.range_.scalarCount);
  }

  Eigen::VectorXd local_x;
  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/SolveLocalSystem/Solve");
    local_x = local_solver_.solve(local_b);
  }

  if (local_solver_.info() != Eigen::Success)
    throw std::runtime_error("failed to solve FEM CPU backend local system");

  {
    KSK_PROFILE_SCOPE("FEMCPUBackend/SolveLocalSystem/ScatterSolution");
    for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
      x[i] = local_x[i];
    }
  }
}

void FEMCPUBackend::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq, runtime::GeometryView globalDx) const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/MapLocalDirectionToGeometry");
  requireCPU(localDq, "FEMCPUBackend::mapLocalDirectionToGeometry");
  requireCPU(globalDx, "FEMCPUBackend::mapLocalDirectionToGeometry");
  if (subsystem_.geometry_points_.size() != subsystem_.samples_.size()) {
    throw std::runtime_error("FEM geometry point mapping is stale");
  }

  for (int sample_index = 0;
       sample_index < static_cast<int>(subsystem_.samples_.size());
       ++sample_index) {
    const runtime::PointIdx point = subsystem_.geometry_points_[sample_index];
    if (point < 0 || point >= globalDx.pointCount()) {
      throw std::invalid_argument(
          "FEM global geometry direction buffer is too small");
    }
    const int offset = subsystem_.samples_[sample_index].qOffset;
    globalDx[point] = glm::dvec3(localDq[offset + 0], localDq[offset + 1],
                                 localDq[offset + 2]);
  }
}

void FEMCPUBackend::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient, runtime::DofView g) const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/ScatterContactGradient");
  requireCPU(pointGradient, "FEMCPUBackend::scatterContactGradient");
  requireCPU(g, "FEMCPUBackend::scatterContactGradient");
  if (points.size() != static_cast<size_t>(pointGradient.pointCount())) {
    throw std::invalid_argument("contact point and gradient counts differ");
  }
  requireDofCount(g, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::scatterContactGradient");

  for (size_t i = 0; i < points.size(); i++) {
    const auto it = std::find(subsystem_.geometry_points_.begin(),
                              subsystem_.geometry_points_.end(), points[i]);
    if (it == subsystem_.geometry_points_.end()) continue;

    const int sample_index = it - subsystem_.geometry_points_.begin();
    const int offset = subsystem_.samples_[sample_index].qOffset;
    g[offset + 0] += pointGradient[i].x;
    g[offset + 1] += pointGradient[i].y;
    g[offset + 2] += pointGradient[i].z;
  }
}

void FEMCPUBackend::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const {
  requireCPU(pointHessianProduct,
             "FEMCPUBackend::applyContactGeometryHessianProduct");
  requireCPU(localY, "FEMCPUBackend::applyContactGeometryHessianProduct");

  scatterContactGradient(productPoints, pointHessianProduct, localY);
}

void FEMCPUBackend::applyInternalContactHessian(runtime::ConstDofView localDq,
                                                runtime::DofView localY) const {
  requireCPU(localDq, "FEMCPUBackend::applyInternalContactHessian");
  requireCPU(localY, "FEMCPUBackend::applyInternalContactHessian");
  requireDofCount(localDq, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::applyInternalContactHessian");
  requireDofCount(localY, subsystem_.range_.scalarCount,
                  "FEMCPUBackend::applyInternalContactHessian");

  for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_) {
    applyContactHessianProduct(subsystem_, contact, localDq, localY);
  }
}

Eigen::VectorXd FEMCPUBackend::buildMassDiagonal() const {
  KSK_PROFILE_SCOPE("FEMCPUBackend/BuildMassDiagonal");

  Eigen::VectorXd mass = Eigen::VectorXd::Zero(subsystem_.range_.scalarCount);

  for (const FEMPrimitive& primitive : subsystem_.primitives()) {
    std::visit(
        [&](const auto& value) {
          using Primitive = std::decay_t<decltype(value)>;
          const int base = value.offset.q;
          if constexpr (std::is_same_v<Primitive, FEMTetMeshPrimitive>) {
            const double vertex_mass = value.mesh.material.density;
            for (int vertex = 0; vertex < value.mesh.vertices.size();
                 vertex++) {
              mass[base + 3 * vertex + 0] = vertex_mass;
              mass[base + 3 * vertex + 1] = vertex_mass;
              mass[base + 3 * vertex + 2] = vertex_mass;
            }
          } else {
            std::vector<double> vertex_mass(value.mesh.vertices.size(), 0.0);
            for (const auto& triangle : value.mesh.triangles) {
              const double face_mass =
                  triangleArea(value.mesh.vertices, triangle) *
                  value.mesh.material.arealDensity / 3.0;

              for (int vertex : triangle) vertex_mass.at(vertex) += face_mass;
            }
            for (int vertex = 0; vertex < value.mesh.vertices.size();
                 vertex++) {
              const double lumped = vertex_mass[vertex];
              mass[base + 3 * vertex + 0] = lumped;
              mass[base + 3 * vertex + 1] = lumped;
              mass[base + 3 * vertex + 2] = lumped;
            }
          }
        },
        primitive);
  }

  return mass;
}
}  // namespace ksk::engine::fem
