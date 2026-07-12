# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(ABD_TOPLEVEL_PROJECT OFF)
else()
    set(ABD_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
if(ABD_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build ABD is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ABDOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/ABDOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/ABDOptions.cmake)
endif()

################################################################################

project(Inukshuk
        DESCRIPTION "Robust, intersection-free, simulations of rigid bodies."
        LANGUAGES CXX CUDA)

option(ABD_WITH_UNIT_TESTS            "Build unit-tests"      ${ABD_TOPLEVEL_PROJECT})
option(ABD_WITH_OPENGL                "Build GUI"             ${ABD_TOPLEVEL_PROJECT})
option(ABD_WITH_TOOLS                 "Build tools"           ${ABD_TOPLEVEL_PROJECT})
option(ABD_WITH_PROFILING             "Profile functions"                          ON)
option(ABD_WITH_COMPARISONS           "Build comparisons"                          OFF)
option(ABD_WITH_SIMD                  "Enable SIMD"                                OFF)
option(ABD_WITH_PYTHON                "Build Python bindings"                      OFF)
option(ABD_WITH_DERIVATIVE_CHECK      "Check derivatives using finite differences" OFF)
option(IPC_TOOLKIT_WITH_CUDA          "Enable CUDA CCD"                            ON)

# Set default minimum C++ standard
if(ABD_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/ABD/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_SEPARABLE_COMPILATION ON)

# Color output
include(abd_use_colors)

include(abd_autogen)

mark_as_advanced(CLEAR CMAKE_BUILD_TYPE)

################################################################################
# External project settings
################################################################################

# libigl options
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_PREDICATES    "Use exact predicates"          ON)
set(LIBIGL_WITH_OPENGL_GLFW_IMGUI ${ABD_WITH_OPENGL} CACHE BOOL "Use OpenGL" FORCE)
set(LIBIGL_WITH_OPENGL_GLFW       ${ABD_WITH_OPENGL} CACHE BOOL "Use GLFW"   FORCE)
set(LIBIGL_WITH_OPENGL            ${ABD_WITH_OPENGL} CACHE BOOL "Use IMGUI"  FORCE)
set(LIBIGL_WITH_PNG               ${ABD_WITH_OPENGL} CACHE BOOL "Use PNG"    FORCE)

# PolySolve settings
option(POLYSOLVE_WITH_CHOLMOD          "Enable Cholmod library"            ON)
option(POLYSOLVE_WITH_UMFPACK          "Enable UmfPack library"            OFF)
option(POLYSOLVE_WITH_SUPERLU          "Enable SuperLU library"            OFF)
option(POLYSOLVE_WITH_MKL              "Enable MKL library"                OFF)
option(POLYSOLVE_WITH_PARDISO          "Enable Pardiso library"            OFF)
option(POLYSOLVE_WITH_HYPRE            "Enable hypre"                      OFF)
option(POLYSOLVE_WITH_AMGCL            "Use AMGCL"                         OFF)
option(POLYSOLVE_WITH_SPECTRA          "Enable computing spectrum"         OFF)

set(TIGHT_INCLUSION_WITH_NO_ZERO_TOI OFF CACHE BOOL "Enable refinement if CCD produces a zero ToI" FORCE)

################################################################################
# ABD Library
################################################################################

add_library(abd
  src/autodiff/autodiff.cpp

  src/utils/sinc.cpp

  src/ccd/accd/accd.hpp
  src/ccd/impact.cpp
  src/ccd/ccd.cpp
  src/ccd/linear/broad_phase.cpp
  # src/ccd/piecewise_linear/time_of_impact.cpp
  src/interval/filib_rounding.cpp
  src/interval/interval_root_finder.cpp
  src/ccd/rigid/broad_phase.cpp
  # src/ccd/rigid/rigid_body_hash_grid.cpp
  src/ccd/rigid/rigid_body_bvh.cpp
  src/ccd/rigid/time_of_impact.cpp
  src/ccd/rigid/rigid_trajectory_aabb.cpp
  # src/ccd/redon/time_of_impact.cpp
  src/ccd/save_queries.cpp

  # src/geometry/intersection.cpp

  src/io/serialize_json.cpp
  src/io/read_rb_scene.cpp
  src/io/read_obj.cpp
  src/io/write_obj.cpp
  src/io/write_gltf.cpp

  src/physics/mass.cpp
  src/utils/mesh_selector.cpp
  src/physics/rigid_body.cpp
  src/physics/rigid_body_assembler.cpp
  src/physics/rigid_body_problem.cpp

  src/opt/distance_barrier_constraint.cpp
  src/opt/collision_constraint.cpp
  src/opt/optimization_problem.cpp
  src/opt/optimization_results.cpp

  src/problems/barrier_problem.cpp

  src/problems/linear_constraint.cpp
  src/problems/linear_constraint_handler.cpp

  src/solvers/newton_solver.cpp
  src/solvers/ipc_solver.cpp
  # src/solvers/line_search.cpp

  src/problems/rigid_body_collision_constraint.cpp
  src/problems/distance_barrier_rb_problem.cpp  
  src/problems/problem_factory.cpp  

  src/utils/tensor.cpp
  src/utils/eigen_ext.cpp
  src/utils/regular_2d_grid.cpp
  src/utils/get_rss.cpp
  src/utils/row_echelon.cpp

  src/SimState.cpp
  src/logger.cpp
  src/profiler.cpp
)
target_include_directories(abd PUBLIC src)
add_library(abd::abd ALIAS abd)

# Autogenerated files ----------------------------------------------------------
# project_autogen(abd time_of_impact_coeff.py time_of_impact_coeff.tpp auto_time_of_impact_coeff .ipp)

################################################################################
# Optional Definitions
################################################################################

# Directory paths
set(ABD_MESH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
target_compile_definitions(abd PUBLIC ABD_MESHES_DIR=\"${ABD_MESH_DIR}\")

if(ABD_WITH_DERIVATIVE_CHECK)
  target_compile_definitions(abd PUBLIC ABD_WITH_DERIVATIVE_CHECK)
endif()

if(ABD_WITH_PROFILING)
  message(STATUS "Profiling Enabled")
  target_compile_definitions(abd PUBLIC ABD_PROFILE_FUNCTIONS)
endif()

# For MSVC, do not use the min and max macros.
target_compile_definitions(abd PUBLIC NOMINMAX)

################################################################################
# Required Libraries
################################################################################

# Extra warnings
include(abd_warnings)
target_link_libraries(abd PUBLIC abd::abd::warnings)

# libigl
include(eigen)
include(libigl)
target_link_libraries(abd PUBLIC
  Eigen3::Eigen
  igl::core
  igl::predicates
)

# JSON Parser
include(json)
target_link_libraries(abd PUBLIC nlohmann_json::nlohmann_json)

# Logger
include(spdlog)
target_link_libraries(abd PUBLIC spdlog::spdlog)

# Finite Differences
include(finite_diff)
target_link_libraries(abd PUBLIC finitediff::finitediff)

# Boost
include(boost)
target_link_libraries(abd PUBLIC Boost::boost)

# TBB
include(onetbb)
target_link_libraries(abd PUBLIC TBB::tbb)

# MSCCD
include(tight_inclusion)
target_link_libraries(abd PUBLIC tight_inclusion::tight_inclusion)

# PolySolve for wrapping linear solvers
include(polysolve)
target_link_libraries(abd PUBLIC PolyFEM::polysolve)

# IPC Toolkit
include(ipc_toolkit)
target_link_libraries(abd PUBLIC ipc::toolkit)

# filib
include(filib)
target_link_libraries(abd PUBLIC filib::filib)

# SimpleBVH
include(simple_bvh)
target_link_libraries(abd PUBLIC simple_bvh::simple_bvh)

# tinygltf
include(tinygltf)
target_link_libraries(abd PUBLIC tinygltf::tinygltf)

# Filesystem
include(filesystem)
target_link_libraries(abd PUBLIC ghc::filesystem)

################################################################################
# Compiler options
################################################################################

## SIMD support
if(ABD_WITH_SIMD)
  # Figure out SSE level support
  message(STATUS "Seaching for SSE...")
  find_package(SSE)
  # Apple broke AVX instructions with clang:
  # https://gitlab.com/libeigen/eigen/-/commit/71aa53dd6dfdc497324d9e87f59c4ba820191856
  if(NOT APPLE)
    # Figure out AVX level support
    message(STATUS "Searching for AVX...")
    find_package(AVX)
    # Figure out FMA level support
    message(STATUS "Searching for FMA...")
    find_package(FMA)
  else()
    message(WARNING "Disabling AVX support: clang compiler shipped with XCode 11.[012] generates broken assembly with -macosx-version-min=10.15 and AVX enabled.")
  endif()
  # Add SSE, AVX, and FMA flags to compiler flags
  string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
  target_compile_options(abd PUBLIC ${SIMD_FLAGS})
endif()

# Use C++17
target_compile_features(abd PUBLIC cxx_std_17)

################################################################################
# Simulations Executable
################################################################################

add_executable(abd_sim src/main.cpp)
set_property(TARGET abd_sim PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(ABD_WITH_OPENGL)
    target_sources(abd_sim PUBLIC
      src/viewer/imgui_ext.cpp
      src/viewer/igl_viewer_ext.cpp
      src/viewer/UISimState.cpp
      src/viewer/UIMenu.cpp
    )
endif()

target_link_libraries(abd_sim PUBLIC abd::abd)

include(cli11)
target_link_libraries(abd_sim PUBLIC CLI11::CLI11)

if(ABD_WITH_OPENGL)
    target_compile_definitions(abd_sim PUBLIC ABD_WITH_OPENGL)

    include(libigl)
    target_link_libraries(abd_sim PUBLIC
        igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::png)

    # Charlie Tangora's gif-h library
    include(gif_h)
    target_link_libraries(abd_sim PRIVATE gif_h::gif_h)
endif()

################################################################################
# Tools
################################################################################
if(ABD_WITH_TOOLS)
  add_subdirectory(src/tools)
endif()

################################################################################
# Tests
################################################################################

# Enable unit testing at the root level
if(ABD_WITH_UNIT_TESTS)
    include(CTest)
    enable_testing()

    # Include Catch2 and provide function `catch_discover_tests` to register tests.
    include(catch2)
    FetchContent_GetProperties(catch2)
    include("${catch2_SOURCE_DIR}/contrib/Catch.cmake")

    add_subdirectory(tests)
endif()

################################################################################
# Comparisons to other methods
################################################################################
if(ABD_WITH_COMPARISONS)
  add_subdirectory(comparisons)
endif()

################################################################################
# Python bindings
################################################################################

if(ABD_WITH_PYTHON)
  add_subdirectory(python)
endif()
