//
// Created by creeper on 5/23/24.
//
#include <fem/integrator.h>
#include <fem/ipc/implicit-euler.h>
#include <ogl-render/ogl-gui.h>
#include <cxxopts.hpp>
using namespace fem;
using namespace opengl;

void checkArgs(const cxxopts::ParseResult &result) {
  if (!result.count("input")) {
    std::cerr << "Please specify input file" << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  cxxopts::Options options("IPC FEM", "FEM Soft body simulator using IPC");
  options.add_options()
      ("i,input", "Input file", cxxopts::value<std::string>())
      ("d,dHat", "dHat", cxxopts::value<Real>()->default_value("1e-3"))
      ("e,eps", "eps", cxxopts::value<Real>()->default_value("1e-6"));
  auto result = options.parse(argc, argv);
  checkArgs(result);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  std::unique_ptr<System> system{};
  auto gui = std::make_unique<OpenglGui>(GuiOption{1024, 1024, "IPC FEM"});
  std::unique_ptr<Integrator> integrator = std::make_unique<IpcImplicitEuler>(*system, IpcIntegrator::Config{
      .dHat = result["d"].as<Real>(),
      .eps = result["e"].as<Real>()
  });
  gui->render([&]() {

  });
}