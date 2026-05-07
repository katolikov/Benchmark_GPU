cmake_minimum_required(VERSION 3.22)
project(bench_gpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# When building for Android, ANDROID_ABI is arm64-v8a and we build the on-device
# benchmarks. When ANDROID_ABI is empty we build the host-only `report` tool.
set(BUILD_FOR_ANDROID FALSE)
if(DEFINED ANDROID_ABI AND NOT "${ANDROID_ABI}" STREQUAL "")
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "Only arm64-v8a is configured (got ${ANDROID_ABI})")
    endif()
    set(BUILD_FOR_ANDROID TRUE)
endif()

set(LIBS ${CMAKE_SOURCE_DIR}/../libs)

add_compile_options(-O3 -Wall -Wno-attributes)
if(BUILD_FOR_ANDROID)
    add_compile_options(-ffast-math -fno-finite-math-only)
    add_link_options(-Wl,--no-undefined -Wl,-z,now)
endif()

# ---------- report --------------------------------------------------------
# Small, no native deps. Built for whatever target we configured for.
add_executable(report report.cpp)
if(NOT BUILD_FOR_ANDROID)
    return()
endif()
target_link_options(report PRIVATE "-Wl,-rpath,\\$ORIGIN")

# ---------- bench_mnn -------------------------------------------------------
add_library(MNN          SHARED IMPORTED)
add_library(MNN_Express  SHARED IMPORTED)
add_library(MNN_CL       SHARED IMPORTED)
add_library(MNN_Vulkan   SHARED IMPORTED)
set_target_properties(MNN          PROPERTIES IMPORTED_LOCATION ${LIBS}/mnn/arm64-v8a/libMNN.so)
set_target_properties(MNN_Express  PROPERTIES IMPORTED_LOCATION ${LIBS}/mnn/arm64-v8a/libMNN_Express.so)
set_target_properties(MNN_CL       PROPERTIES IMPORTED_LOCATION ${LIBS}/mnn/arm64-v8a/libMNN_CL.so)
set_target_properties(MNN_Vulkan   PROPERTIES IMPORTED_LOCATION ${LIBS}/mnn/arm64-v8a/libMNN_Vulkan.so)

add_executable(bench_mnn bench_mnn.cpp)
target_include_directories(bench_mnn PRIVATE ${LIBS}/mnn/include ${CMAKE_SOURCE_DIR})
# Link only libMNN (others are dlopen'd at runtime by libMNN itself when the
# matching forward type is requested), but keep them on the search path so the
# loader can find them at runtime.
target_link_libraries(bench_mnn PRIVATE MNN MNN_Express log)
target_link_options(bench_mnn PRIVATE "-Wl,-rpath,\\$ORIGIN")

# ---------- bench_tflite ----------------------------------------------------
add_library(tflite_jni     SHARED IMPORTED)
add_library(tflite_gpu_jni SHARED IMPORTED)
set_target_properties(tflite_jni
    PROPERTIES IMPORTED_LOCATION ${LIBS}/tflite/arm64-v8a/libtensorflowlite_jni.so)
set_target_properties(tflite_gpu_jni
    PROPERTIES IMPORTED_LOCATION ${LIBS}/tflite/arm64-v8a/libtensorflowlite_gpu_jni.so)

add_executable(bench_tflite bench_tflite.cpp)
target_include_directories(bench_tflite PRIVATE ${LIBS}/tflite/include ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_tflite PRIVATE tflite_jni tflite_gpu_jni log)
target_link_options(bench_tflite PRIVATE "-Wl,-rpath,\\$ORIGIN")
