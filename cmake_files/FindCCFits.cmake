# - Try to find CCFits framework
if(CCFits_FOUND)
  return()
endif()

find_path(CCFits_INCLUDE_DIR CCfits.h HINT ${EXTERNAL_ROOT}/include)

set(CCFits_INCLUDE_DIRS ${CCFits_INCLUDE_DIR} )

find_library(CCFits_LIBRARY NAMES CCfits)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CCFits  DEFAULT_MSG CCFits_INCLUDE_DIR CCFits_LIBRARY)
