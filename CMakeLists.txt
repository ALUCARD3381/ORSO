cmake_minimum_required(VERSION 3.18)

project(ORSO VERSION 0.7.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ORSO_ENABLE_NEON "Enable ARM NEON kernels when supported" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)
# ------------------------------------------------------------
# NEON detection
# ------------------------------------------------------------

include(CheckCXXSourceCompiles)

set(ORSO_USE_NEON OFF)

if(ORSO_ENABLE_NEON)
    set(ORSO_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS
        "${CMAKE_REQUIRED_FLAGS} -mfpu=neon"
    )

    check_cxx_source_compiles("
        #include <arm_neon.h>

        int main() {
            float32x4_t v = vdupq_n_f32(1.0f);
            (void)v;
            return 0;
        }
    " ORSO_NEON_COMPILES)

    set(CMAKE_REQUIRED_FLAGS "${ORSO_SAVED_REQUIRED_FLAGS}")

    if(ORSO_NEON_COMPILES)
        set(ORSO_USE_NEON ON)
        message(STATUS "ORSO: NEON kernels enabled")
    else()
        message(STATUS "ORSO: NEON not available; using scalar fallback")
    endif()
endif()

# ------------------------------------------------------------
# Core implementation
# ------------------------------------------------------------

add_library(orso_core_impl STATIC
    core/src/tensor.cpp
    core/src/neon_kernels.cpp
    core/src/transformer.cpp
    core/src/model.cpp
    core/src/optimizer.cpp
)

target_include_directories(orso_core_impl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

target_compile_features(orso_core_impl PUBLIC cxx_std_20)

if(ORSO_USE_NEON)
    target_compile_definitions(
        orso_core_impl
        PUBLIC
            ORSO_USE_NEON=1
    )

    target_compile_options(
        orso_core_impl
        PUBLIC
            -mfpu=neon
    )
endif()

# ------------------------------------------------------------
# Python / pybind11 module
# ------------------------------------------------------------

pybind11_add_module(orso_core
    bindings/bindings.cpp
)

target_link_libraries(
    orso_core
    PRIVATE
        orso_core_impl
)

target_include_directories(
    orso_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

target_compile_features(
    orso_core
    PRIVATE
        cxx_std_20
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

set_target_properties(
    orso_core
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ------------------------------------------------------------
# Build information
# ------------------------------------------------------------

message(STATUS "----------------------------------------")
message(STATUS "ORSO version: ${PROJECT_VERSION}")
message(STATUS "C++ standard: C++20")
message(STATUS "Python: ${Python3_EXECUTABLE}")

if(ORSO_USE_NEON)
    message(STATUS "ORSO SIMD: ARM NEON")
else()
    message(STATUS "ORSO SIMD: scalar fallback")
endif()

message(STATUS "----------------------------------------")
