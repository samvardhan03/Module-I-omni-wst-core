cmake_minimum_required(VERSION 3.18)
project(omni_wst_core LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Detect GPU architecture or default to sm_80
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(SOURCES
  cpp/wst_bindings.cu
  cpp/memory_staging.cu
)

pybind11_add_module(_core MODULE ${SOURCES})

target_include_directories(_core PRIVATE cpp/)
target_link_libraries(_core PRIVATE
  CUDA::cudart
  CUDA::cufft
)

target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

install(TARGETS _core DESTINATION omni_wst_core)
