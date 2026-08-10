#include <Runtime/subsystem.h>

#include <Runtime/newton-ipc-subsystem-api.h>
#include <Runtime/projective-dynamics-subsystem-api.h>

namespace ksk::runtime {

// Default factory implementations: a subsystem that does not support a solver
// family returns nullptr. Defined out-of-line so the returned std::unique_ptr
// has the complete solver API type available at destruction time.
std::unique_ptr<NewtonIPCSubsystemSolver> Subsystem::createNewtonIPCSolver()
{
  return nullptr;
}

std::unique_ptr<ProjectiveDynamicsSubsystemSolver>
Subsystem::createProjectiveDynamicsSolver()
{
  return nullptr;
}

}  // namespace ksk::runtime
