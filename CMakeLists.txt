cmake_minimum_required(VERSION 3.16)
project(ORSO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(orso_core
    bindings/bindings.cpp
    core/src/tensor.cpp
)

target_include_directories(orso_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

target_compile_features(orso_core PRIVATE cxx_std_20)

target_compile_options(orso_core PRIVATE
    -O2
    -Wall
    -Wextra
    -Wpedantic
)
