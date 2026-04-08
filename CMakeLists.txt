cmake_minimum_required(VERSION 3.18)
project(adas_perception_stack VERSION 1.0.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# C++ standard & global compile options
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
)

add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)

# ---------------------------------------------------------------------------
# Find external dependencies (optional – guarded so the project configures
# even without a full GPU / ONNX Runtime install)
# ---------------------------------------------------------------------------
find_package(CUDA QUIET)
find_package(Threads REQUIRED)

# Eigen3 (required – install via: apt install libeigen3-dev  or  brew install eigen)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# ONNX Runtime  (set ONNXRUNTIME_ROOT or let CMake find it via pkg-config)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS $ENV{ONNXRUNTIME_ROOT}/include /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS $ENV{ONNXRUNTIME_ROOT}/lib /opt/onnxruntime/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
  message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
  set(HAVE_ONNXRUNTIME TRUE)
  add_compile_definitions(HAVE_ONNXRUNTIME)
else()
  message(WARNING "ONNX Runtime not found – inference will use stub implementation")
  set(HAVE_ONNXRUNTIME FALSE)
endif()

# TensorRT  (optional)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIB nvinfer
  HINTS $ENV{TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
  message(STATUS "TensorRT found: ${TENSORRT_LIB}")
  set(HAVE_TENSORRT TRUE)
  add_compile_definitions(HAVE_TENSORRT)
else()
  message(WARNING "TensorRT not found – TRT path will be disabled")
  set(HAVE_TENSORRT FALSE)
endif()

# ---------------------------------------------------------------------------
# Collect sources
# ---------------------------------------------------------------------------
set(COMMON_SOURCES
  src/common/logger.cpp
)

set(INFERENCE_SOURCES
  src/inference/onnx_runner.cpp
  src/inference/tensorrt_runner.cpp
)

set(PERCEPTION_SOURCES
  src/perception/object_detector.cpp
  src/perception/lane_detector.cpp
  src/perception/sensor_fusion.cpp
)

set(TRACKING_SOURCES
  src/tracking/multi_object_tracker.cpp
)

set(PREDICTION_SOURCES
  src/prediction/trajectory_predictor.cpp
)

set(PLANNING_SOURCES
  src/planning/acc_controller.cpp
  src/planning/aeb_controller.cpp
  src/planning/lane_keeping_assist.cpp
  src/planning/traffic_light_handler.cpp
)
if(CUDA_FOUND)
  set(CUDA_SOURCES
    src/cuda/nms_kernel.cu
    src/cuda/preprocess_kernel.cu
  )
endif()

# ---------------------------------------------------------------------------
# Core library  (linkable target reused by main + tests)
# ---------------------------------------------------------------------------
add_library(adas_core STATIC
  ${COMMON_SOURCES}
  ${INFERENCE_SOURCES}
  ${PERCEPTION_SOURCES}
  ${TRACKING_SOURCES}
  ${PREDICTION_SOURCES}
  ${PLANNING_SOURCES}
  $<$<BOOL:${CUDA_FOUND}>:${CUDA_SOURCES}>
)

target_include_directories(adas_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<$<BOOL:${HAVE_ONNXRUNTIME}>:${ONNXRUNTIME_INCLUDE_DIR}>
  $<$<BOOL:${HAVE_TENSORRT}>:${TENSORRT_INCLUDE_DIR}>
  $<$<BOOL:${CUDA_FOUND}>:${CUDA_INCLUDE_DIRS}>
)

target_link_libraries(adas_core PUBLIC
  Threads::Threads
  Eigen3::Eigen
  $<$<BOOL:${HAVE_ONNXRUNTIME}>:${ONNXRUNTIME_LIB}>
  $<$<BOOL:${HAVE_TENSORRT}>:${TENSORRT_LIB}>
  $<$<BOOL:${CUDA_FOUND}>:${CUDA_LIBRARIES}>
)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(adas_demo src/main.cpp)
target_link_libraries(adas_demo PRIVATE adas_core)

# ---------------------------------------------------------------------------
# Unit tests (Google Test)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(TARGETS adas_demo DESTINATION bin)
install(TARGETS adas_core DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
