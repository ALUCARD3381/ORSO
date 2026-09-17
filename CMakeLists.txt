cmake_minimum_required(VERSION 3.16)
project(ORSO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ORSO_ENABLE_NEON "Enable ARMv7 NEON kernels when building on 32-bit ARM" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

set(ORSO_SOURCES
    bindings/bindings.cpp
    core/src/tensor.cpp
    core/src/neon_kernels.cpp
)

pybind11_add_module(orso_core ${ORSO_SOURCES})

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

# ARMv7/32-bit ARM: request NEON explicitly. A scalar fallback remains available.
if(ORSO_ENABLE_NEON AND CMAKE_SIZEOF_VOID_P EQUAL 4 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM|armv7|armv7l)")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mfpu=neon" ORSO_COMPILER_HAS_NEON)
    if(ORSO_COMPILER_HAS_NEON)
        target_compile_options(orso_core PRIVATE -mfpu=neon)
        target_compile_definitions(orso_core PRIVATE ORSO_ENABLE_NEON=1)
        message(STATUS "ORSO: NEON kernels enabled")
    else()
        message(WARNING "ORSO: compiler does not accept -mfpu=neon; using scalar fallback")
    endif()
else()
    message(STATUS "ORSO: scalar kernels (NEON disabled/not applicable)")
endif()
