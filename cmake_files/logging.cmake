# Setup logging
set(PURIPSI_LOGGER_NAME "puripsi" CACHE STRING "NAME of the logger")
set(PURIPSI_COLOR_LOGGING true CACHE BOOL "Whether to add color to the log")
if(logging)
  set(PURIPSI_DO_LOGGING 1)
  set(PURIPSI_TEST_LOG_LEVEL critical CACHE STRING "Level when logging tests")
  set_property(CACHE PURIPSI_TEST_LOG_LEVEL PROPERTY STRINGS
    off critical error warn info debug trace)
else()
  unset(PURIPSI_DO_LOGGING)
  set(PURIPSI_TEST_LOG_LEVEL off)
endif()

