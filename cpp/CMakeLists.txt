# adds include dirs which are located within the build directory
# That is, whathever package is downloaded automatically by the build system
# This complication arises because cmake won't install targets with headers that point to the build
# dir.
# Users will have to install packages globally for all the right headers to be found
function(add_include_dir)
  unset(privates)
  foreach(dir ${ARGN})
    file(RELATIVE_PATH rel "${PROJECT_BINARY_DIR}" "${dir}" )
    if(NOT "${rel}" MATCHES "^\\.\\./")
      list(APPEND privates "${dir}")
    endif()
  endforeach()
  if(privates)
    include_directories(SYSTEM ${privates})
  endif()
endfunction()

add_include_dir(
  ${PSI_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CFitsIO_INCLUDE_DIR}
  ${CCFits_INCLUDE_DIR}/..
  ${CImg_INCLUDE_DIR}
  )

add_subdirectory(puripsi)
if(tests OR examples OR benchmarks)
  configure_file(tests/directories.in.h "${PROJECT_BINARY_DIR}/include/puripsi/directories.h")
endif()
if(examples)
  add_subdirectory(example)
endif()
if(tests)
  configure_file(tests/data.in.h "${PROJECT_BINARY_DIR}/include/puripsi/test_data.h")
  add_subdirectory(tests)
endif()
if(benchmarks)
  add_subdirectory(benchmarks)
endif()

if(TARGET casacore::ms)
  add_executable(puripsi main.cc cmdl.cc)
  target_link_libraries(puripsi libpuripsi)
  set_target_properties(puripsi PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

  install(TARGETS puripsi
    EXPORT PuriPsiTargets
    DESTINATION share/cmake/PuriPsi
    RUNTIME DESTINATION bin
    )
endif()
