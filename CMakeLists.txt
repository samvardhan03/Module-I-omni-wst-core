cmake_minimum_required(VERSION 3.24)
project(omni-wst-core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Link against cudart and cufft
find_package(CUDAToolkit REQUIRED)

option(COMMERCIAL_USE "Enable commercial enterprise features" OFF)

if(COMMERCIAL_USE)
    add_compile_definitions(COMMERCIAL_USE=1)
endif()

pybind11_add_module(_core MODULE 
    cpp/wst_bindings.cpp 
    cpp/memory_staging.cu
)

target_include_directories(_core PRIVATE cpp)

target_link_libraries(_core PRIVATE 
    CUDA::cudart
    CUDA::cufft
)

# Install into the src/omni_wst_core directory
install(TARGETS _core DESTINATION omni_wst_core)
