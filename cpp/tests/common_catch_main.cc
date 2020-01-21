#define CATCH_CONFIG_RUNNER

#include "puripsi/config.h"
#include <psi/config.h>
#include <catch2/catch.hpp>
#include <memory>
#include <random>
#include <psi/logging.h>
#include "puripsi/logging.h"

std::unique_ptr<std::mt19937_64> mersenne(new std::mt19937_64(0));

int main(int argc, char **argv) {
  Catch::Session session; // There must be exactly once instance

  int returnCode = session.applyCommandLine(argc, const_cast<char **>(argv));
  if(returnCode != 0) // Indicates a command line error
    return returnCode;
  mersenne.reset(new std::mt19937_64(session.configData().rngSeed));

  psi::logging::initialize();
  puripsi::logging::initialize();

  return session.run();
}
