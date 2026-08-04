cmake_minimum_required(VERSION 3.18)

project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
file(GLOB HEADER_FILES ../sadl/*.h )

include_directories(..)
add_executable(test_scalar  test.cpp ${HEADER_FILES})
add_executable(test_avx2    test.cpp ${HEADER_FILES})
add_executable(test_avx512  test.cpp ${HEADER_FILES})
add_executable(test_avx2_mempool    test.cpp ${HEADER_FILES})
add_executable(test_avx2_sparse    test.cpp ${HEADER_FILES})
add_executable(test_sadl_int16_scalar  test_sadl_int16.cpp ${HEADER_FILES})
add_executable(test_sadl_int16_avx2    test_sadl_int16.cpp ${HEADER_FILES})
add_executable(test_sadl_int16_avx512  test_sadl_int16.cpp ${HEADER_FILES})
add_executable(test_sadl_int16_avx2_mempool  test_sadl_int16.cpp ${HEADER_FILES})
add_executable(test_sadl_int16_avx2_sparse   test_sadl_int16.cpp ${HEADER_FILES})

if( UNIX OR MINGW )
 set(CMAKE_CXX_FLAGS_RELEASE "-O3")
 set(CMAKE_CXX_FLAGS "-ffast-math -Wall -fstrict-aliasing") 
 set_target_properties(test_scalar PROPERTIES COMPILE_FLAGS "" )
 set_target_properties(test_avx2   PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1" )
 set_target_properties(test_avx512 PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -DNDEBUG=1" )
 set_target_properties(test_avx2_mempool PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1 -DSADL_USE_MEMORY_POOL=1" )
 set_target_properties(test_avx2_sparse PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1 -DSPARSE_SUPPORT=1")
 set_target_properties(test_sadl_int16_scalar PROPERTIES COMPILE_FLAGS "" )
 set_target_properties(test_sadl_int16_avx2 PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1" )
 set_target_properties(test_sadl_int16_avx512 PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -DNDEBUG=1" )
 set_target_properties(test_sadl_int16_avx2_mempool PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1 -DSADL_USE_MEMORY_POOL=1" )
 set_target_properties(test_sadl_int16_avx2_sparse PROPERTIES COMPILE_FLAGS "-mavx2 -DNDEBUG=1 -DSPARSE_SUPPORT=1")
endif()


if( MSVC )
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  set_target_properties(test_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_target_properties(test_sadl_int16_avx2 PROPERTIES COMPILE_FLAGS "/arch:AVX2")
  set_target_properties(test_sadl_int16_avx512 PROPERTIES COMPILE_FLAGS "/arch:AVX512")
  set_target_properties(test_sadl_int16_avx2_mempool PROPERTIES COMPILE_FLAGS "/arch:AVX2 /DSADL_USE_MEMORY_POOL=1")
  set_target_properties(test_sadl_int16_avx2_sparse PROPERTIES COMPILE_FLAGS "/arch:AVX2 /DSPARSE_SUPPORT=1")
  string( APPEND CMAKE_CXX_FLAGS " /WX" ) #treat warnings as errors
endif()

add_executable(test_cpu_int16 test_cpu_int16.cpp)
target_include_directories(test_cpu_int16 PRIVATE ..)
if( UNIX OR MINGW )
  set_target_properties(test_cpu_int16 PROPERTIES COMPILE_FLAGS "-O3 -DNDEBUG=1")
endif()

cmake_policy(SET CMP0104 NEW)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  find_package(CUDAToolkit REQUIRED)

  add_executable(test_cuda_int16 test_cuda_int16.cu)
  target_include_directories(test_cuda_int16 PRIVATE ..)
  target_link_libraries(test_cuda_int16 PRIVATE CUDA::cudart)
  set_target_properties(test_cuda_int16 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES          native
  )
  target_compile_options(test_cuda_int16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
  )

  message(STATUS "CUDA found: test_cuda_int16 will be built")
else()
  message(WARNING "CUDA compiler not found: test_cuda_int16 will NOT be built")
endif()

