include(AddExample)

include_directories(SYSTEM ${PSI_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/cpp")
include_directories(SYSTEM ${CASAREST_INCLUDE_DIRS} REQUIRED)

add_example(gridding LIBRARIES libpuripsi NOTEST)
add_example(generate_vis_data LIBRARIES libpuripsi NOTEST)

add_example(show_image LIBRARIES libpuripsi NOTEST)

add_example(wideband_blocking_read_and_write_uv_data_serial LIBRARIES libpuripsi NOTEST)
add_example(wideband_blocking_read_and_write_uv_data_parallel LIBRARIES libpuripsi NOTEST)
add_example(wideband_blocking_load_uv_data_serial LIBRARIES libpuripsi NOTEST)

add_example(primal_dual_time_blocking LIBRARIES libpuripsi NOTEST )
add_example(primal_dual_time_blocking_multi_ms LIBRARIES libpuripsi NOTEST )
add_example(primal_dual_time_blocking_random_coverage LIBRARIES libpuripsi NOTEST )

add_example(preconditioned_primaldual_random_coverage LIBRARIES libpuripsi NOTEST )

add_example(wideband_primal_dual_time_blocking LIBRARIES libpuripsi NOTEST )

add_example(wideband_primal_dual_time_blocking_ms LIBRARIES libpuripsi NOTEST )

add_example(load_interleaved_bands LIBRARIES libpuripsi NOTEST)

add_example(script_read_MS LIBRARIES libpuripsi NOTEST)

add_example(wideband_primal_dual_time_blocking_random_coverage LIBRARIES libpuripsi NOTEST )


if(hdf5)
  include_directories(${HDF5_INCLUDE_DIRS})
  add_example(read_visibilities_h5 LIBRARIES libpuripsi ${HDF5_LIBRARIES} NOTEST)
  add_example(wideband_primal_dual_time_blocking_h5 LIBRARIES libpuripsi ${HDF5_LIBRARIES} NOTEST)
endif()

if(data AND TARGET casacore::ms)
  add_example(casa LIBRARIES libpuripsi)
endif()
