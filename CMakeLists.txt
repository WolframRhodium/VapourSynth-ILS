cmake_minimum_required(VERSION 3.20)

project(VapourSynth-ILS LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDAToolkit REQUIRED)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

add_library(vs_ils OBJECT 
    source/vs_ils.cpp
    source/win32.cpp
)

target_include_directories(vs_ils PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(vs_ils PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_library(ils_impl OBJECT source/ils_impl.cu)
set_target_properties(ils_impl PROPERTIES
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_include_directories(ils_impl PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
)

add_library(ils SHARED)
target_link_libraries(ils vs_ils ils_impl CUDA::cudart_static CUDA::cufft)

if (MSVC)
    target_link_options(ils PRIVATE "/DELAYLOAD:cufft64_10.dll" "delayimp.lib")
endif()

