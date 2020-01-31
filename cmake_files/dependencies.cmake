# Scripts to run puripsi from build directory. Good for testing/debuggin.
include(EnvironmentScript)
# Look up packages: if not found, installs them

# Look for external software
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(Threads)
  if(THREADS_FOUND)
    add_compile_options(-pthread)
  endif(THREADS_FOUND)
endif()

# Always find open-mp, since it may be used by psi
find_package(OpenMP)
if(OPENMP_FOUND)
  add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)
  set_target_properties(openmp::openmp PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
    INTERFACE_LINK_LIBRARIES  "${OpenMP_CXX_FLAGS}")
endif()
if(openmp AND NOT OPENMP_FOUND)
    message(STATUS "Could not find OpenMP. Compiling without.")
endif()
set(PURIPSI_OPENMP_FFTW FALSE)
if(openmp AND OPENMP_FOUND)
  set(PURIPSI_OPENMP TRUE)
  find_package(FFTW3 REQUIRED DOUBLE SERIAL COMPONENTS OPENMP)
  set(FFTW3_DOUBLE_LIBRARY fftw3::double::serial)
  if(TARGET fftw3::double::openmp)
    list(APPEND FFTW3_DOUBLE_LIBRARY fftw3::double::openmp)
    set(PURIPSI_OPENMP_FFTW TRUE)
  endif()
else()
  set(PURIPSI_OPENMP FALSE)
  find_package(FFTW3 REQUIRED DOUBLE)
  set(FFTW3_DOUBLE_LIBRARY fftw3::double::serial)
endif()

set(PURIPSI_MPI FALSE)
if(mpi)
  find_package(MPI REQUIRED)
  set(PURIPSI_MPI TRUE)
endif()


find_package(TIFF REQUIRED)


if(data AND tests)
  find_package(Boost REQUIRED COMPONENTS filesystem)
else()
  find_package(Boost REQUIRED)
endif()

find_package(Eigen3 REQUIRED)

if(logging)
  find_package(spdlog REQUIRED)
endif()

# Look up packages: if not found, installs them
# Unless otherwise specified, if puripsi is not on master, then psi will be
# downloaded from development branch.
if(NOT PSI_GIT_TAG)
  set(PSI_GIT_TAG master CACHE STRING "Branch/tag when downloading psi")
endif()
if(NOT PSI_GIT_REPOSITORY)
  set(PSI_GIT_REPOSITORY https://www.github.com/basp-group/psi-dev.git
    CACHE STRING "Location when downloading psi")
endif()
if(mpi)
  find_package(Psi REQUIRED COMPONENTS mpi ARGUMENTS
    GIT_REPOSITORY ${PSI_GIT_REPOSITORY}
    GIT_TAG ${PSI_GIT_TAG}
    MPI "TRUE")
else()
  find_package(Psi REQUIRED ARGUMENTS
    GIT_REPOSITORY ${PSI_GIT_REPOSITORY}
    GIT_TAG ${PSI_GIT_TAG})
endif()

  
find_package(CFitsIO REQUIRED ARGUMENTS CHECKCASA)
find_package(CCFits REQUIRED)

find_package(CasaCore OPTIONAL_COMPONENTS ms)

find_package(CasaRest REQUIRED)

set(PURIPSI_EIGEN_MKL FALSE)
# Find MKL
find_package(MKL)

if(MKL_FOUND AND mkl)
    set(PURIPSI_EIGEN_MKL 1) # This will go into config.h
    set(EIGEN_USE_MKL_ALL 1) # This will go into config.h - it makes Eigen use MKL
    include_directories(${MKL_INCLUDE_DIR})
else()
    set(PURIPSI_EIGEN_MKL 0)
    set(EIGEN_USE_MKL_ALL 0)
endif()


# Add script to execute to make sure libraries in the build tree can be found
add_to_ld_path("${EXTERNAL_ROOT}/lib")
