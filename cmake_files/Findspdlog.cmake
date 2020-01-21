# - Try to find spdlog framework
if(spdlog_FOUND)
  return()
endif()

find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h HINT ${EXTERNAL_ROOT}/include)

set(SPDLOG_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR} )

find_library(SPDLOG_LIBRARY NAMES spdlog)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(spdlog  DEFAULT_MSG SPDLOG_INCLUDE_DIR SPDLOG_LIBRARY)
