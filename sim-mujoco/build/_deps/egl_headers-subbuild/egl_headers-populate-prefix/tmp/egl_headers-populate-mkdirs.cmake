# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-src")
  file(MAKE_DIRECTORY "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-src")
endif()
file(MAKE_DIRECTORY
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-build"
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix"
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/tmp"
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/src/egl_headers-populate-stamp"
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/src"
  "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/src/egl_headers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/src/egl_headers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/metemadi/dev/robot-control-stack/build/_deps/egl_headers-subbuild/egl_headers-populate-prefix/src/egl_headers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
