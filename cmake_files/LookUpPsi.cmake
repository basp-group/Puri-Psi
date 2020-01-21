# Looks up [Psi](http://basp-group.github.io/psi/)
#
# - GIT_REPOSITORY: defaults to https://github.com/basp-group/psi-dev.git
# - GIT_TAG: defaults to master
# - BUILD_TYPE: defaults to Release
#
if(PSI_ARGUMENTS)
    cmake_parse_arguments(Psi "" "GIT_REPOSITORY;GIT_TAG;BUILD_TYPE" ""
        ${PSI_ARGUMENTS})
endif()
if(NOT PSI_GIT_REPOSITORY)
    set(PSI_GIT_REPOSITORY https://github.com/basp-group/psi-dev.git)
endif()
if(NOT PSI_GIT_TAG)
    set(PSI_GIT_TAG master)
endif()
if(NOT PSI_BUILD_TYPE)
  set(PSI_BUILD_TYPE Release)
endif()
if(NOT PSI_MPI)
  set(PSI_MPI OFF)
endif()

# WARNING THIS IS FOR TESTING THE DEVELOPMENT BRANCH.
# THIS SHOULD BE REMOVED BEFORE THE BRANCH IS MERGED
set(PSI_GIT_TAG development CACHE  STRING "…" FORCE)

# write subset of variables to cache for psi to use
include(PassonVariables)
passon_variables(Lookup-Psi
  FILENAME "${EXTERNAL_ROOT}/src/PsiVariables.cmake"
  PATTERNS
      "CMAKE_[^_]*_R?PATH" "CMAKE_C_.*"
      "BLAS_.*" "FFTW3_.*" "TIFF_.*"
      "GreatCMakeCookOff_DIR"
  ALSOADD
      "\nset(CMAKE_INSTALL_PREFIX \"${EXTERNAL_ROOT}\" CACHE STRING \"\")\n"
)
ExternalProject_Add(
    Lookup-Psi
    PREFIX ${EXTERNAL_ROOT}
    GIT_REPOSITORY ${PSI_GIT_REPOSITORY}
    GIT_TAG ${PSI_GIT_TAG}
    CMAKE_ARGS
      -C "${EXTERNAL_ROOT}/src/PsiVariables.cmake"
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_BUILD_TYPE=${PSI_BUILD_TYPE}
      -Dregressions=OFF
      -Dtests=OFF
      -Dpython=OFF
      -Dexamples=OFF
      -Dbenchmarks=OFF
      -Dlogging=${logging}
      -DNOEXPORT=TRUE
      -Dopenmp=${openmp}
      -Dmpi=${PSI_MPI)
    INSTALL_DIR ${EXTERNAL_ROOT}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)
add_recursive_cmake_step(Lookup-Psi DEPENDEES install)

foreach(dep Lookup-Eigen3 Lookup-spdlog)
  find_package(${dep})
  if(TARGET ${dep})
    add_dependencies(Lookup-Psi ${dep})
  endif()
endforeach()
