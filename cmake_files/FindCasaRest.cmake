macro(casarest_resolve_dependencies _result)
  set(${_result} ${ARGN})
  set(_index 0)
  # Do a breadth-first search through the dependency graph; append to the
  # result list the dependent components for each item in that list. 
  # Duplicates will be removed later.
  while(1)
    list(LENGTH ${_result} _length)
    if(NOT _index LESS _length)
      break()
    endif(NOT _index LESS _length)
    list(GET ${_result} ${_index} item)
    list(APPEND ${_result} ${Casarest_${item}_DEPENDENCIES})
    math(EXPR _index "${_index}+1")
  endwhile(1)
  # Remove all duplicates in the current result list, while retaining only the
  # last of each duplicate.
  list(REVERSE ${_result})
  list(REMOVE_DUPLICATES ${_result})
  list(REVERSE ${_result})
endmacro(casarest_resolve_dependencies _result)


# - casarest_find_library(_name)
#
# Search for the library ${_name}. 
# If library is found, add it to CASAREST_LIBRARIES; if not, add ${_name}
# to CASAREST_MISSING_COMPONENTS and set CASAREST_FOUND to false.
#
#   Usage: casarest_find_library(name)
#
macro(casarest_find_library _name)
  string(TOUPPER ${_name} _NAME)
  find_library(${_NAME}_LIBRARY ${_name}
    HINTS ${CASAREST_ROOT_DIR} PATH_SUFFIXES lib)
  mark_as_advanced(${_NAME}_LIBRARY)
  if(${_NAME}_LIBRARY)
    list(APPEND CASAREST_LIBRARIES ${${_NAME}_LIBRARY})
  else(${_NAME}_LIBRARY)
    set(CASAREST_FOUND FALSE)
    list(APPEND CASAREST_MISSING_COMPONENTS ${_name})
  endif(${_NAME}_LIBRARY)
endmacro(casarest_find_library _name)


# - casarest_find_package(_name)
#
# Search for the package ${_name}.
# If the package is found, add the contents of ${_name}_INCLUDE_DIRS to
# CASAREST_INCLUDE_DIRS and ${_name}_LIBRARIES to CASAREST_LIBRARIES.
#
# If Casarest itself is required, then, strictly speaking, the packages it
# requires must be present. However, when linking against static libraries
# they may not be needed. One can override the REQUIRED setting by switching
# CASAREST_MAKE_REQUIRED_EXTERNALS_OPTIONAL to ON. Beware that this might cause
# compile and/or link errors.
#
#   Usage: casarest_find_package(name [REQUIRED])
#
macro(casarest_find_package _name)
  if("${ARGN}" MATCHES "^REQUIRED$" AND
      Casarest_FIND_REQUIRED AND
      NOT CASAREST_MAKE_REQUIRED_EXTERNALS_OPTIONAL)
    find_package(${_name} REQUIRED)
  else()
    find_package(${_name})
  endif()
  if(${_name}_FOUND)
    list(APPEND CASAREST_INCLUDE_DIRS ${${_name}_INCLUDE_DIRS})
    list(APPEND CASAREST_LIBRARIES ${${_name}_LIBRARIES})
  endif(${_name}_FOUND)
endmacro(casarest_find_package _name)


# Define the Casarest components.
set(Casarest_components
  msvis
  calibration
  synthesis
#  flagging
#  simulators
)

# Define the Casarest components' inter-dependencies.
set(Casarest_calibration_DEPENDENCIES  msvis)
set(Casarest_synthesis_DEPENDENCIES calibration)
#set(Casarest_flagging_DEPENDENCIES  msvis)

# Initialize variables.
set(CASAREST_FOUND FALSE)
set(CASAREST_DEFINITIONS)
set(CASAREST_LIBRARIES)
set(CASAREST_MISSING_COMPONENTS)

# Search for the header file first. Note that casarest installs the header
# files in ${prefix}/include/casarest, instead of ${prefix}/include.
if(NOT CASAREST_INCLUDE_DIR)
  find_path(CASAREST_INCLUDE_DIR msvis/MSVis/VisSet.h
    HINTS ${CASAREST_ROOT_DIR} PATH_SUFFIXES include/casarest)
  mark_as_advanced(CASAREST_INCLUDE_DIR)
endif(NOT CASAREST_INCLUDE_DIR)

if(NOT CASAREST_INCLUDE_DIR)
  set(CASAREST_ERROR_MESSAGE "Casarest: unable to find the header file msvis/MSVis/VisSet.h.\nPlease set CASAREST_ROOT_DIR to the root directory containing Casarest.")
else()
  # We've found the header file; let's continue.
  set(CASAREST_FOUND TRUE)
  set(CASAREST_INCLUDE_DIRS ${CASAREST_INCLUDE_DIR})
  list(APPEND CASAREST_INCLUDE_DIRS ${CASAREST_INCLUDE_DIR}/casarest)
  # If the user specified components explicity, use that list; otherwise we'll
  # assume that the user wants to use all components.
  if(NOT Casarest_FIND_COMPONENTS)
    set(Casarest_FIND_COMPONENTS ${Casarest_components})
  endif(NOT Casarest_FIND_COMPONENTS)

  # Get a list of all dependent Casarest libraries that need to be found.
  casarest_resolve_dependencies(_find_components ${Casarest_FIND_COMPONENTS})

  # Find the library for each component, and handle external dependencies
  foreach(_comp ${_find_components})
    casarest_find_library(casa_${_comp})
  endforeach(_comp ${_find_components})

endif(NOT CASAREST_INCLUDE_DIR)

# Set HAVE_CASAREST
if(CASAREST_FOUND)
  set(HAVE_CASAREST TRUE CACHE INTERNAL "Define if Casarest is installed")
endif(CASAREST_FOUND)

# Compose diagnostic message if not all necessary components were found.
if(CASAREST_MISSING_COMPONENTS)
  set(CASAREST_ERROR_MESSAGE "Casarest: the following components could not be found:\n     ${CASAREST_MISSING_COMPONENTS}")
endif(CASAREST_MISSING_COMPONENTS)

# Print diagnostics and add to intefaces
if(CASAREST_FOUND)
  if(NOT Casarest_FIND_QUIETLY)
    message(STATUS "Found the following Casarest components: ")
    foreach(comp ${_find_components})
      message(STATUS "  ${comp}")
    endforeach(comp ${_find_components})
    list(APPEND INTERFACE_INCLUDE_DIRECTORIES "${CASAREST_INCLUDE_DIR}")
  endif(NOT Casarest_FIND_QUIETLY)
else(CASAREST_FOUND)
  if(Casarest_FIND_REQUIRED)
    message(FATAL_ERROR "${CASAREST_ERROR_MESSAGE}")
  else(Casarest_FIND_REQUIRED)
    message(STATUS "${CASAREST_ERROR_MESSAGE}")
  endif(Casarest_FIND_REQUIRED)
endif(CASAREST_FOUND)
