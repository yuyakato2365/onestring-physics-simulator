if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit    
	GIT_REPOSITORY https://github.com/Ahdhn/ipc-toolkit.git
	GIT_TAG 3ada3a78b9212bf6c3bff0381f54acc44e35891a
	#GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
	#GIT_TAG d41d6ca93cf9b9c4c01a3177fbb68b63c6b74df1
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)

# ONESTRING: CUDA 13 requires MSVC's conforming preprocessor. The Abseil
# revision pinned by this legacy IPC Toolkit uses qualified injected-class
# names that the conforming preprocessor rejects. Apply the two equivalent
# modern spellings before the dependency is compiled.
if(IPC_TOOLKIT_WITH_CUDA)
    FetchContent_GetProperties(abseil SOURCE_DIR abseil_SOURCE_DIR)
endif()
if(IPC_TOOLKIT_WITH_CUDA AND abseil_SOURCE_DIR)
    set(_onestring_absl_hash_internal
        "${abseil_SOURCE_DIR}/absl/hash/internal/hash.h")
    set(_onestring_absl_hash "${abseil_SOURCE_DIR}/absl/hash/hash.h")
    if(EXISTS "${_onestring_absl_hash_internal}")
        file(READ "${_onestring_absl_hash_internal}" _onestring_absl_contents)
        string(REPLACE
            "friend class MixingHashState::HashStateBase;"
            "friend class HashStateBase<MixingHashState>;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        string(REPLACE
            "friend class HashStateBase;"
            "friend class HashStateBase<MixingHashState>;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        string(REPLACE
            "using MixingHashState::HashStateBase::combine_contiguous;"
            "using HashStateBase<MixingHashState>::combine_contiguous;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        file(WRITE "${_onestring_absl_hash_internal}" "${_onestring_absl_contents}")
    endif()
    if(EXISTS "${_onestring_absl_hash}")
        file(READ "${_onestring_absl_hash}" _onestring_absl_contents)
        string(REPLACE
            "using HashState::HashStateBase::combine_contiguous;"
            "using hash_internal::HashStateBase<HashState>::combine_contiguous;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        string(REPLACE
            "using HashStateBase::combine_contiguous;"
            "using hash_internal::HashStateBase<HashState>::combine_contiguous;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        string(REPLACE
            "friend class HashState::HashStateBase;"
            "friend class hash_internal::HashStateBase<HashState>;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        string(REPLACE
            "friend class HashStateBase;"
            "friend class hash_internal::HashStateBase<HashState>;"
            _onestring_absl_contents "${_onestring_absl_contents}")
        file(WRITE "${_onestring_absl_hash}" "${_onestring_absl_contents}")
    endif()
endif()
