# ATHENA patched CMakeLists.txt for whisper.cpp examples/talk-llama.
#
# Differences from upstream (both load-bearing — keep when syncing with upstream):
#   1. ONNX Runtime block: -DONNXRUNTIME_ROOT=<dir> enables the emotion2vec speech-
#      emotion tagger (defines ATHENA_EMOTION_ORT=1, links libonnxruntime, bakes the
#      lib dir into RPATH). Unset -> tagging compiles out, byte-identical baseline.
#   2. Conditional DeepSeek Sparse Attention source: current whisper.cpp syncs ship
#      llama-kv-cache-dsa.cpp/.h in this example; the ATHENA-pinned commit
#      (afa2ea544fb4) predates it. The EXISTS guard makes this one file build BOTH
#      trees — on a current pull the DSA translation unit is compiled (omitting it
#      breaks the link), on the pinned tree the guard no-ops.
#
# The ATHENA-custom sources (talk-llama.cpp, athena_memory.h, silero-endpointer.h,
# silero-turn-state.h) need no listing here: only .cpp files are compiled directly,
# and the three headers are #included by talk-llama.cpp from the same directory.

if (WHISPER_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    file(GLOB SRC_MODELS models/*.cpp)

    # Newer llama syncs only (DeepSeek Sparse Attention KV cache) — see header note.
    set(SRC_DSA "")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/llama-kv-cache-dsa.cpp)
        set(SRC_DSA llama-kv-cache-dsa.cpp)
    endif()

    set(TARGET whisper-talk-llama)
    add_executable(${TARGET} talk-llama.cpp
        llama.cpp
        llama-adapter.cpp
        llama-arch.cpp
        llama-batch.cpp
        llama-chat.cpp
        llama-context.cpp
        llama-cparams.cpp
        llama-grammar.cpp
        llama-graph.cpp
        llama-hparams.cpp
        llama-impl.cpp
        llama-io.cpp
        llama-kv-cache.cpp
        llama-kv-cache-iswa.cpp
        llama-memory-recurrent.cpp
        llama-memory-hybrid.cpp
        llama-memory-hybrid-iswa.cpp
        llama-memory.cpp
        llama-mmap.cpp
        llama-model-loader.cpp
        llama-model-saver.cpp
        llama-model.cpp
        llama-quant.cpp
        llama-sampler.cpp
        llama-vocab.cpp
        unicode.cpp
        unicode-data.cpp
        ${SRC_DSA}
        ${SRC_MODELS})
    target_include_directories(${TARGET} PRIVATE . ${SDL2_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${SDL2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    # ────────────────────────────────────────────────────────────────────────
    # emotion2vec SER via ONNX Runtime (optional). Configure with:
    #   -DONNXRUNTIME_ROOT=<ATHENA>/onnxruntime-linux-x64-gpu_cuda12-1.27.0
    # If unset, whisper-talk-llama builds exactly as before (tagging compiled out).
    if(ONNXRUNTIME_ROOT)
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
        target_include_directories(${TARGET} PRIVATE ${ONNXRUNTIME_ROOT}/include)
        target_link_libraries(${TARGET} PRIVATE ${ONNXRUNTIME_LIB})
        target_compile_definitions(${TARGET} PRIVATE ATHENA_EMOTION_ORT=1)
        # bake the ORT lib dir into RPATH so the .so is found at runtime
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH   ${ONNXRUNTIME_ROOT}/lib
            INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
        message(STATUS "talk-llama: emotion2vec ONNX tagging ENABLED (${ONNXRUNTIME_LIB})")
    else()
        message(STATUS "talk-llama: ONNXRUNTIME_ROOT unset -> emotion tagging disabled")
    endif()

    install(TARGETS ${TARGET} RUNTIME)

    if(WIN32)
        # It requires Windows 8.1 or later for PrefetchVirtualMemory
        target_compile_definitions(${TARGET} PRIVATE -D_WIN32_WINNT=0x0602)
    endif()

    include(DefaultTargetOptions)
endif ()
