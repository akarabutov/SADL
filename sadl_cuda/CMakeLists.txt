cmake_minimum_required(VERSION 3.18)
project(sadl_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)


add_library(sadl_cuda INTERFACE)
target_include_directories(sadl_cuda INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..      
)
target_link_libraries(sadl_cuda INTERFACE CUDA::cudart)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sample/sample_cuda.cu")
    add_executable(sample_cuda ../sample/sample_cuda.cu)
    target_link_libraries(sample_cuda PRIVATE sadl_cuda)
    set_target_properties(sample_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(UNIX OR MINGW)
    target_compile_options(sadl_cuda INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    )
endif()
