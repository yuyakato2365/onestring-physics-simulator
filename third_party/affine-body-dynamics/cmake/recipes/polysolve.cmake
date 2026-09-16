if(TARGET PolyFEM::polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'PolyFEM::polysolve'")

include(FetchContent)

# ONESTRING: libigl's legacy DownloadProject helper is also used by the pinned
# PolySolve checkout.  With CMake 4 it must receive an explicit compatibility
# policy. Nested download-only projects must not receive the CUDA toolset a
# second time; recent CMake already propagates it and rejects duplicate -T.
if(DEFINED libigl_SOURCE_DIR)
    set(_onestring_download_project "${libigl_SOURCE_DIR}/cmake/DownloadProject.cmake")
    if(EXISTS "${_onestring_download_project}")
        file(READ "${_onestring_download_project}" _onestring_download_contents)
        if(_onestring_download_contents MATCHES "ONESTRING_CMAKE4_CHILD")
            string(REPLACE
                "                        -T \"\${CMAKE_GENERATOR_TOOLSET}\"\n"
                ""
                _onestring_download_contents "${_onestring_download_contents}")
            file(WRITE "${_onestring_download_project}" "${_onestring_download_contents}")
        else()
            set(_onestring_child_prefix "# ONESTRING_CMAKE4_CHILD\n    execute_process(COMMAND \${CMAKE_COMMAND} -G \"\${CMAKE_GENERATOR}\"\n                        -D \"CMAKE_POLICY_VERSION_MINIMUM:STRING=3.5\"")
            string(REPLACE
                "execute_process(COMMAND \${CMAKE_COMMAND} -G \"\${CMAKE_GENERATOR}\""
                "${_onestring_child_prefix}"
                _onestring_download_contents "${_onestring_download_contents}")
            file(WRITE "${_onestring_download_project}" "${_onestring_download_contents}")
        endif()
    endif()
endif()

FetchContent_Declare(
    polysolve
    GIT_REPOSITORY https://github.com/polyfem/polysolve.git
    GIT_TAG 72e5eaca17b1ae975fa5a7149627a17e6b13cf80
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)

add_library(PolyFEM::polysolve ALIAS polysolve)
