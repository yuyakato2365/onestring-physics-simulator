if(TARGET software_renderer::software_renderer)
    return()
endif()

message(STATUS "Third-party: creating target 'software_renderer::software_renderer'")

include(FetchContent)
FetchContent_Declare(
    software_renderer
    GIT_REPOSITORY https://github.com/Ahdhn/software-renderer.git
    GIT_TAG 924fc6cd4830ee7fbfb92c02ebb95a8f2750b957
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(software_renderer)
