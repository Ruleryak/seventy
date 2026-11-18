cmake_minimum_required(VERSION 3.26)
project(seventy_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70-real)
option(SEVENTY_LINEINFO "Add line information for profilers" OFF)

find_package(CUDAToolkit 12.0 REQUIRED)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
  message(FATAL_ERROR "CUDA ${CUDAToolkit_VERSION} cannot compile for sm70. Use CUDA 12.x (12.9 recommended).")
endif()

file(GLOB SEVENTY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu)
add_library(seventy_kernels SHARED ${SEVENTY_SOURCES})
target_include_directories(seventy_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set_target_properties(seventy_kernels PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(seventy_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas=-warn-spills>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SEVENTY_LINEINFO}>>:-lineinfo>)
target_link_options(seventy_kernels PRIVATE -Wl,--no-undefined)

install(TARGETS seventy_kernels LIBRARY DESTINATION seventy/_lib)
