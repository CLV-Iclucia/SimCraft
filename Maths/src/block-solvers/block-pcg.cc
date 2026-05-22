#include <Maths/block-solvers/block-pcg.h>

// Note: BlockPCGSolver does not go through the legacy LinearSolver factory.
// This file exists for future unified factory registration.
// Currently IpcIntegrator::create() directly constructs the block solver.

namespace sim::maths {
// Reserved for future factory registration if needed.
} // namespace sim::maths
