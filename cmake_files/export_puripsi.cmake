# Exports PuriPsi so other packages can access it
set(targets libpuripsi)
if(TARGET puripsi)
  list(APPEND targets puripsi)
endif()
export(TARGETS ${targets} FILE "${PROJECT_BINARY_DIR}/PuriPsiTargets.cmake")

# Avoids creating an entry in the cmake registry.
if(NOT NOEXPORT)
    export(PACKAGE PuriPsi)
endif()

# First in binary dir
set(ALL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp" "${PROJECT_BINARY_DIR}/include")
configure_File(cmake_files/PuriPsiConfig.in.cmake
    "${PROJECT_BINARY_DIR}/PuriPsiConfig.cmake" @ONLY
)
configure_File(cmake_files/PuriPsiConfigVersion.in.cmake
    "${PROJECT_BINARY_DIR}/PuriPsiConfigVersion.cmake" @ONLY
)

# Then for installation tree
file(RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/share/cmake/puripsi"
    "${CMAKE_INSTALL_PREFIX}/include"
)
set(ALL_INCLUDE_DIRS "\${PURIPSI_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake_files/PuriPsiConfig.in.cmake
    "${PROJECT_BINARY_DIR}/CMakeFiles/PuriPsiConfig.cmake" @ONLY
)

# Finally install all files
install(FILES
    "${PROJECT_BINARY_DIR}/CMakeFiles/PuriPsiConfig.cmake"
    "${PROJECT_BINARY_DIR}/PuriPsiConfigVersion.cmake"
    DESTINATION share/cmake/puripsi
    COMPONENT dev
)

install(EXPORT PuriPsiTargets DESTINATION share/cmake/puripsi COMPONENT dev)
