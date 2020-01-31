#ifndef PURIPSI_CPP_CONFIG_H
#define PURIPSI_CPP_CONFIG_H

//! Problems with using and constructors
#cmakedefine PURIPSI_HAS_USING
#ifndef PURIPSI_HAS_USING
#define PURIPSI_HAS_NOT_USING
#endif

//! Whether to do logging or not
#cmakedefine PURIPSI_DO_LOGGING

//! Whether to do openmp
#cmakedefine PURIPSI_OPENMP

//! Whether FFTW has openmp
#cmakedefine PURIPSI_OPENMP_FFTW

//! Whether mpi is being used or not
#cmakedefine PURIPSI_MPI

// Whether Eigen will use MKL (if MKL was found and PURIPSI_EIGEN_MKL is enabled in CMake)
#cmakedefine PURIPSI_EIGEN_MKL
#cmakedefine EIGEN_USE_MKL_ALL

#include <string>
#include <tuple>

namespace puripsi {
//! Returns library version
inline std::string version() { return "@PURIPSI_VERSION@"; }
//! Returns library version
inline std::tuple<uint8_t, uint8_t, uint8_t> version_tuple() {
  // clang-format off
  return std::tuple<uint8_t, uint8_t, uint8_t>(
      @PURIPSI_VERSION_MAJOR@, @PURIPSI_VERSION_MINOR@, @PURIPSI_VERSION_PATCH@);
  // clang-format on
}
//! Returns library git reference, if known
inline std::string gitref() { return "@PURIPSI_GITREF@"; }
//! Default logging level
inline std::string default_logging_level() { return "@PURIPSI_TEST_LOG_LEVEL@"; }
//! Default logger name
inline std::string default_logger_name() { return "@PURIPSI_LOGGER_NAME@"; }
//! Wether to add color to the logger
inline constexpr bool color_logger() {
  // clang-format off
  return @PURIPSI_COLOR_LOGGING@;
  // clang-format on
}
}

#endif
