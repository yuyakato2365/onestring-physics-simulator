
# ONESTRING: short FetchContent base
# Keep nested dependency clones away from Windows path-length limits.
if(IPC_TOOLKIT_WITH_CUDA)
    set(_onestring_fetchcontent_dir "C:/os_fc_cuda")
else()
    set(_onestring_fetchcontent_dir "C:/os_fc")
endif()
set(
    FETCHCONTENT_BASE_DIR
    "${_onestring_fetchcontent_dir}"
    CACHE PATH
    "Short directory for downloaded CMake dependencies"
    FORCE
)
