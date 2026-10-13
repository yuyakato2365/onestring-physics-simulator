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
# Compatibility with CMake >=4, where the official upstream 3.1 minimum
# is no longer accepted. This changes only CMake's policy interpretation,
# not any source file in the BFF checkout.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
    "Minimum compatibility policy for the legacy upstream BFF project" FORCE)
add_subdirectory("${ONESTRING_BFF_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/bff-upstream-build")

# On modern Homebrew, SuiteSparse libraries can be visible under
# /opt/homebrew/lib while the old upstream FindSuiteSparse.cmake leaves
# SuiteSparse_CHOLMOD_INCLUDE_DIR-NOTFOUND. Locate the actual cholmod.h
# independently and attach its directory to the official BFF target.
# Do NOT mutate the official C++ sources or use a fake BFF backend.
set(_onestring_suitesparse_prefixes "")
if(APPLE)
    execute_process(
        COMMAND brew --prefix suite-sparse
        OUTPUT_VARIABLE _onestring_brew_suitesparse_prefix
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_onestring_brew_suitesparse_prefix)
        list(APPEND _onestring_suitesparse_prefixes "${_onestring_brew_suitesparse_prefix}")
    endif()
    list(APPEND _onestring_suitesparse_prefixes "/opt/homebrew" "/usr/local")
endif()

find_path(ONESTRING_CHOLMOD_HEADER_DIR
    NAMES cholmod.h
    HINTS ${_onestring_suitesparse_prefixes}
    PATH_SUFFIXES include/suitesparse include SuiteSparse suitesparse
)
if(NOT ONESTRING_CHOLMOD_HEADER_DIR)
    message(FATAL_ERROR
        "Official BFF requires SuiteSparse's cholmod.h. "
        "The original FindSuiteSparse.cmake may find libcholmod.dylib "
        "without finding its headers. Check: "
        "find $(brew --prefix suite-sparse) -name cholmod.h -print. "
        "If missing, install with: brew install suite-sparse. "
        "If present in a custom prefix, pass "
        "-DONESTRING_CHOLMOD_HEADER_DIR=/path/to/containing/directory."
    )
endif()
message(STATUS "OneString official BFF CHOLMOD headers: ${ONESTRING_CHOLMOD_HEADER_DIR}")
target_include_directories(bff PUBLIC "${ONESTRING_CHOLMOD_HEADER_DIR}")

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
