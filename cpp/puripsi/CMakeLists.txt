# Add to install includes only if outside build dir
function(add_include_dir)
  unset(publics)
  unset(privates)
  foreach(dir ${ARGN})
    file(RELATIVE_PATH rel "${PROJECT_BINARY_DIR}" "${dir}" )
    if("${rel}" MATCHES "^\\.\\./")
      list(APPEND publics "${dir}")
    endif()
  endforeach()
  if(publics)
    target_include_directories(libpuripsi SYSTEM PUBLIC ${publics})
  endif()
endfunction()



configure_file(config.in.h "${PROJECT_BINARY_DIR}/include/puripsi/config.h")

set(HEADERS
  logging.h kernels.h
  pfitsio.h logging.disabled.h types.h 
  index_mapping.h logging.enabled.h utilities.h operators.h wproj_utilities.h projection_kernels.h cimg.h
  preconditioner.h time_blocking.h astrodecomposition.h astroio.h "${PROJECT_BINARY_DIR}/include/puripsi/config.h")

set(SOURCES utilities.cc pfitsio.cc
  kernels.cc wproj_utilities.cc operators.cc projection_kernels.cc time_blocking.cc astrodecomposition.cc astroio.cc)

if(TARGET casacore::ms)
  list(APPEND SOURCES casacore.cc)
  list(APPEND HEADERS casacore.h)
endif()

add_library(libpuripsi SHARED ${SOURCES})
set(version "${PURIPSI_VERSION_MAJOR}.${PURIPSI_VERSION_MINOR}.${PURIPSI_VERSION_PATCH}")
set(soversion "${PURIPSI_VERSION_MAJOR}.${PURIPSI_VERSION_MINOR}")
set_target_properties(libpuripsi PROPERTIES VERSION ${version} SOVERSION ${soversion})
set_target_properties(libpuripsi PROPERTIES OUTPUT_NAME puripsi)

target_include_directories(libpuripsi PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
  $<INSTALL_INTERFACE:include/>)


add_include_dir(
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${CFitsIO_INCLUDE_DIR}
  ${CCFits_INCLUDE_DIR}
  )
target_link_libraries(libpuripsi
  ${FFTW3_DOUBLE_LIBRARY} ${CCFits_LIBRARY} ${CFitsIO_LIBRARY} ${PSI_CPP_LIBRARY} ${X11_X11_LIB} ${SPDLOG_LIBRARY})
if(TARGET casacore::casa)
  target_link_libraries(libpuripsi casacore::ms)
endif()
if(TARGET openmp::openmp)
  target_link_libraries(libpuripsi openmp::openmp)
endif()




install(FILES ${HEADERS} DESTINATION include/puripsi)
install(TARGETS libpuripsi
  EXPORT PuriPsiTargets
  DESTINATION share/cmake/PuriPsi
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
  )
