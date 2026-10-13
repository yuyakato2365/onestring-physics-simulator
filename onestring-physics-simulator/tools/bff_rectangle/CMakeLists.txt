cmake_minimum_required(VERSION 3.12)
project(onestring_rectangular_bff LANGUAGES CXX)

if(NOT DEFINED ONESTRING_BFF_SOURCE)
    message(FATAL_ERROR "Pass -DONESTRING_BFF_SOURCE=/absolute/path/to/official/boundary-first-flattening")
endif()
if(NOT EXISTS "${ONESTRING_BFF_SOURCE}/include/bff/project/Bff.h")
    message(FATAL_ERROR "Source root does not contain the official BFF C++ library")
endif()
set(BFF_BUILD_CLI OFF CACHE BOOL "No need to rebuild official CLI" FORCE)
set(BFF_BUILD_GUI OFF CACHE BOOL "Headless adapter" FORCE)
add_subdirectory("${ONESTRING_BFF_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/bff-upstream-build")
# Upstream predates stricter standard-library header dependency enforcement.
# Include the standard uint8_t definition while compiling the original
# library, without patching the user's BFF checkout.
target_compile_options(bff PRIVATE -include cstdint)
add_executable(bff-rect-command-line bff-rect-command-line.cpp)
target_link_libraries(bff-rect-command-line PRIVATE bff)
set_target_properties(bff-rect-command-line PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)
