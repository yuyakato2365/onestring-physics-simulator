if(TARGET Boost::boost)
    return()
endif()

message(STATUS "Third-party: creating targets 'Boost::boost'")

include(FetchContent)
FetchContent_Declare(
    boost-cmake
    GIT_REPOSITORY https://github.com/Orphis/boost-cmake.git
    GIT_TAG 40cb41d86eab0d7fdc18af4b04b733f8cc852d2a
)

set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(OLD_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BOOST_URL "https://archives.boost.io/release/1.71.0/source/boost_1_71_0.tar.bz2" CACHE STRING "Boost download URL")
set(BOOST_URL_SHA256 "d73a8da01e8bf8c7eda40b4c84915071a8c8a0df4a6734537ddde4a8580524ee" CACHE STRING "Boost download URL SHA256 checksum")

# This guy will download boost using FetchContent
FetchContent_GetProperties(boost-cmake)
if(NOT boost-cmake_POPULATED)
    FetchContent_Populate(boost-cmake)
    # File lcid.cpp from Boost_locale.cpp doesn't compile on MSVC, so we exclude them from the default
    # targets being built by the project (only targets explicitly used by other targets will be built).
    add_subdirectory(${boost-cmake_SOURCE_DIR} ${boost-cmake_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ${OLD_CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_CXX_FLAGS "${PREVIOUS_CMAKE_CXX_FLAGS}")
