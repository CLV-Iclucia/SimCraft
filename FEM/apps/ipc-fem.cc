//
// Created by creeper on 5/23/24.
//
#include <cxxopts.hpp>
#include <fem/fem-simulation.h>
#include <iostream>
using namespace sim::fem;

void checkArgs(const cxxopts::ParseResult &result) {
  if (!result.count("input")) {
    std::cerr << "Please specify input file" << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  cxxopts::Options options("IPC FEM", "FEM Soft body simulator using IPC");
  options.add_options()("i,input", "Input file", cxxopts::value<std::string>());
  auto result = options.parse(argc, argv);
  checkArgs(result);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  auto inputFile = result["input"].as<std::string>();
  auto simBuilder = FEMSimulationBuilder{};
  auto simConfig = core::loadJsonFile(inputFile);

  if (!simConfig) {
    std::cerr << "Failed to load simulation configuration from " << inputFile
              << std::endl;
    return 1;
  }

  core::Frame frame;
  auto femSim = simBuilder.build(*simConfig);
  while (frame.idx < 1000) {
    femSim.step(frame);
  }

}