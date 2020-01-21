include_directories("${PROJECT_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_BINARY_DIR}/include" ${GBENCHMARK_INCLUDE_DIR})

add_benchmark(measurement_operator utilities.cc LIBRARIES libpuripsi)
add_benchmark(primaldual utilities.cc LIBRARIES libpuripsi)
