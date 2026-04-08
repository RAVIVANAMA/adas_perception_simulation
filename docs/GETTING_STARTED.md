# 🚀 Getting Started

This guide walks you through setting up and building the ADAS Perception Simulation Stack on **Linux**, **macOS**, and **Windows**.

---

## Table of Contents

1. [Prerequisites Overview](#1-prerequisites-overview)
2. [Linux (Ubuntu / Debian)](#2-linux-ubuntu--debian)
3. [macOS (Homebrew)](#3-macos-homebrew)
4. [Windows (Visual Studio + vcpkg)](#4-windows-visual-studio--vcpkg)
5. [Optional: ONNX Runtime](#5-optional-onnx-runtime)
6. [Optional: TensorRT](#6-optional-tensorrt)
7. [Optional: CUDA Toolkit](#7-optional-cuda-toolkit)
8. [Python Environment](#8-python-environment)
9. [Build Configurations](#9-build-configurations)
10. [Running the Demo](#10-running-the-demo)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites Overview

| Dependency | Minimum Version | Required for C++ build | Notes |
|---|---|---|---|
| CMake | 3.18 | ✅ Yes | [cmake.org](https://cmake.org) |
| C++ compiler | GCC 11 / Clang 14 / MSVC 19.34 | ✅ Yes | C++17 support required |
| Eigen3 | 3.4.0 | ✅ Yes | Header-only linear algebra |
| Ninja | any | ✅ Recommended | Faster builds than make |
| Python | 3.10+ | For visualizer only | |
| CUDA Toolkit | 11.8+ | ⚪ Optional | GPU kernels |
| ONNX Runtime | 1.16+ | ⚪ Optional | DNN inference |
| TensorRT | 8.6+ | ⚪ Optional | TRT acceleration |

> **No GPU required.** The project builds and runs in full stub mode on any CPU-only machine.

---

## 2. Linux (Ubuntu / Debian)

### 2.1 System Packages

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    g++-11 \
    cmake \
    ninja-build \
    libeigen3-dev \
    git \
    python3 \
    python3-pip
```

### 2.2 Clone and Build

```bash
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

# Release build (recommended for performance)
./scripts/build.sh Release

# Or manually:
cmake -S . -B build/Release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release -- -j$(nproc)
```

### 2.3 Run Tests and Demo

```bash
cd build/Release
ctest --output-on-failure

# Run C++ demo pipeline
./adas_demo
```

---

## 3. macOS (Homebrew)

### 3.1 Install Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew update
brew install cmake ninja eigen
```

### 3.2 Build

```bash
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

cmake -S . -B build/Release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR=$(brew --prefix eigen)/share/eigen3/cmake

cmake --build build/Release -- -j$(sysctl -n hw.ncpu)
cd build/Release && ctest --output-on-failure
```

> **Note:** TensorRT and CUDA are not available on macOS. The project will build in stub mode.

---

## 4. Windows (Visual Studio + vcpkg)

### 4.1 Install Tools

1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/) with the **Desktop development with C++** workload.
2. Install [CMake 3.28+](https://cmake.org/download/) — check **"Add to PATH"** during setup.
3. Install [Git for Windows](https://git-scm.com/download/win).

### 4.2 Install Eigen3 via vcpkg (recommended)

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install eigen3:x64-windows
```

### 4.3 Build

```powershell
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

cmake -S . -B build\Release `
      -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build build\Release --config Release
```

### 4.4 Run Tests

```powershell
cd build\Release
ctest -C Release --output-on-failure
.\adas_demo.exe
```

### 4.5 Alternative: MinGW / MSYS2

```bash
# In MSYS2 MinGW64 shell:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
          mingw-w64-x86_64-ninja mingw-w64-x86_64-eigen3

cmake -S . -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release
```

---

## 5. Optional: ONNX Runtime

ONNX Runtime enables real DNN inference (instead of stub data).

### Linux / macOS

```bash
# 1. Download pre-built binary from GitHub releases
#    https://github.com/microsoft/onnxruntime/releases
#    e.g.: onnxruntime-linux-x64-1.17.3.tgz

tar xzf onnxruntime-linux-x64-1.17.3.tgz
export ORT_DIR=$(pwd)/onnxruntime-linux-x64-1.17.3

# 2. Pass to CMake
cmake -S . -B build/Release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -Donnxruntime_DIR=$ORT_DIR/lib/cmake/onnxruntime
```

### Windows (PowerShell)

```powershell
# Download onnxruntime-win-x64-1.17.3.zip and extract to C:\ort

cmake -S . -B build\Release `
      -G "Visual Studio 17 2022" -A x64 `
      -Donnxruntime_DIR=C:\ort\lib\cmake\onnxruntime
```

When CMake detects ONNX Runtime, it sets `-DHAVE_ONNXRUNTIME` and links the library automatically.

---

## 6. Optional: TensorRT

TensorRT requires an NVIDIA GPU and CUDA.

```bash
# Prerequisite: CUDA Toolkit must be installed (see Section 7)
# Install TensorRT from https://developer.nvidia.com/tensorrt
# After installing, set:

cmake -S . -B build/Release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTensorRT_DIR=/path/to/TensorRT-8.6.1.6 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

---

## 7. Optional: CUDA Toolkit

| Platform | Download |
|---|---|
| Ubuntu 22.04 | [CUDA 12.x deb network installer](https://developer.nvidia.com/cuda-downloads) |
| Windows | [CUDA 12.x .exe installer](https://developer.nvidia.com/cuda-downloads) |

```bash
# Ubuntu: after .deb install
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cmake -S . -B build/Release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DHAVE_CUDA=ON
```

---

## 8. Python Environment

```bash
# Visualizer only (minimal)
pip install pygame numpy

# Full tooling (training, export, eval)
pip install -r python/requirements.txt

# Launch visualizer
python python/visualization/adas_visualizer.py
```

See [docs/PYTHON_TOOLING.md](PYTHON_TOOLING.md) for the full training and export guide.

---

## 9. Build Configurations

```bash
# Debug build (assertions, debug symbols, no optimisation)
cmake -S . -B build/Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

# RelWithDebInfo (production + debug symbols — useful for profiling)
cmake -S . -B build/RelWithDebInfo -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/RelWithDebInfo

# Enable ASAN (Address Sanitizer) for memory checks — Debug only
cmake -S . -B build/ASAN -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build/ASAN
./build/ASAN/adas_demo
```

---

## 10. Running the Demo

### C++ Synthetic Pipeline (30 Hz, ~10 seconds)

```bash
./build/Release/adas_demo
# or on Windows:
.\build\Release\adas_demo.exe
```

Expected output:
```
[INFO]  ADAS Pipeline starting — 30 Hz, 300 frames
[INFO]  Frame   0 | Objects: 3 | Tracks: 2 | Control: throttle=0.42 brake=0.00
[INFO]  Frame   1 | Objects: 3 | Tracks: 3 | Control: throttle=0.38 brake=0.00
...
[INFO]  AEB WARNING  | TTC=3.8s | PartialBrake
[INFO]  AEB ACTIVE   | TTC=1.8s | FullBrake
...
[INFO]  Trajectory predictions (5-second snapshot):
        Track 0: [3.2,  1.1] → [4.8,  1.2] → [6.4,  1.2] (CV)
        Track 1: [8.1, -0.3] → [9.9, -0.4] → [11.6, -0.5] (CTRA)
```

### Python Visualizer (real-time GUI)

```bash
python python/visualization/adas_visualizer.py
```

---

## 11. Troubleshooting

### CMake cannot find Eigen3

```
CMake Error: Could not find a package configuration file provided by "Eigen3"
```

**Fix:**
```bash
# Ubuntu
sudo apt install libeigen3-dev

# macOS
brew install eigen

# Manual: download eigen-3.4.0, extract, pass to CMake
cmake ... -DEigen3_DIR=/path/to/eigen-3.4.0/cmake
```

### GoogleTest download fails (no internet access)

```
CMake Error: FetchContent_MakeAvailable: Failed to download googletest
```

**Fix:** Download [googletest-1.14.0.zip](https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip) manually and point `FETCHCONTENT_SOURCE_DIR_googletest` to the extracted directory:
```bash
cmake ... -DFETCHCONTENT_SOURCE_DIR_googletest=/path/to/googletest-1.14.0
```

### CUDA not found despite install

```bash
# Verify nvcc is accessible
which nvcc
nvcc --version

# Set CUDA root explicitly
cmake ... -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Visualizer: `pygame` module not found

```bash
pip install pygame numpy
# If using conda:
conda install -c conda-forge pygame numpy
```

### Linker error: undefined reference to `Ort::*`

Ensure the ORT library directory is on the linker path:
```bash
cmake ... -Donnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```
