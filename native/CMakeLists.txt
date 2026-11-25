cmake_minimum_required(VERSION 3.25)
project(FaultWatchNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
include(FetchContent)
option(FAULTWATCH_BUILD_PYTHON "Build Python inference bindings" ON)

set(FAULTWATCH_ORT_ARCHIVE
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.32.0/microsoft.ml.onnxruntime.1.32.0.nupkg"
    CACHE STRING "URL or local path to the pinned Microsoft ONNX Runtime CPU archive")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
if(APPLE AND processor MATCHES "^(arm64|aarch64)$")
    set(ort_platform osx-arm64)
    set(ort_filename libonnxruntime.dylib)
    set(ort_soname libonnxruntime.1.dylib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(processor MATCHES "^(arm64|aarch64)$")
        set(ort_platform linux-arm64)
    elseif(processor MATCHES "^(x86_64|amd64)$")
        set(ort_platform linux-x64)
    endif()
    set(ort_filename libonnxruntime.so)
    set(ort_soname libonnxruntime.so.1)
endif()
if(NOT ort_platform)
    message(FATAL_ERROR "Supported ONNX Runtime builds: macOS arm64 and Linux arm64/x86-64")
endif()

FetchContent_Declare(onnxruntime
    URL "${FAULTWATCH_ORT_ARCHIVE}"
    URL_HASH SHA256=a9150bc2992cbbb6722070da9910e1ffa989a8bfc67bdd5bb8e745ca755641db
    DOWNLOAD_NAME onnxruntime.zip
)
FetchContent_MakeAvailable(onnxruntime)
set(ort_library "${onnxruntime_SOURCE_DIR}/runtimes/${ort_platform}/native/${ort_filename}")
add_library(onnxruntime_cpu SHARED IMPORTED)
set_target_properties(onnxruntime_cpu PROPERTIES
    IMPORTED_LOCATION "${ort_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/build/native/include"
)
# The package stores the unversioned filename. The loader uses the SONAME.
file(CREATE_LINK "${ort_library}" "${CMAKE_CURRENT_BINARY_DIR}/${ort_soname}" SYMBOLIC)
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")

add_library(faultwatch_engine STATIC src/engine.cpp src/features.cpp)
target_include_directories(faultwatch_engine PUBLIC include)
target_link_libraries(faultwatch_engine PRIVATE onnxruntime_cpu)
target_compile_options(faultwatch_engine PRIVATE -Wall -Wextra -Wpedantic)

add_executable(faultwatch-native src/main.cpp)
target_link_libraries(faultwatch-native PRIVATE faultwatch_engine)

add_executable(faultwatch-native-bench benchmarks/features.cpp)
target_link_libraries(faultwatch-native-bench PRIVATE faultwatch_engine)

if(FAULTWATCH_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    FetchContent_Declare(pybind11
        URL https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v3.0.1
        URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
    )
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(_faultwatch_native src/bindings.cpp)
    target_link_libraries(_faultwatch_native PRIVATE faultwatch_engine)
    set_target_properties(_faultwatch_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
endif()

if(BUILD_TESTING)
    set(BUILD_GMOCK OFF)
    FetchContent_Declare(googletest
        URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    )
    FetchContent_MakeAvailable(googletest)
    add_executable(faultwatch_native_tests tests/inference_test.cpp)
    target_link_libraries(faultwatch_native_tests PRIVATE faultwatch_engine GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(faultwatch_native_tests)
endif()
