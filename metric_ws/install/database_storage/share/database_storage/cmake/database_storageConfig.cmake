# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_database_storage_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED database_storage_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(database_storage_FOUND FALSE)
  elseif(NOT database_storage_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(database_storage_FOUND FALSE)
  endif()
  return()
endif()
set(_database_storage_CONFIG_INCLUDED TRUE)

# output package information
if(NOT database_storage_FIND_QUIETLY)
  message(STATUS "Found database_storage: 0.0.0 (${database_storage_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'database_storage' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT database_storage_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(database_storage_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${database_storage_DIR}/${_extra}")
endforeach()
