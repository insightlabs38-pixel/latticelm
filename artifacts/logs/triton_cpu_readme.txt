# Triton-CPU

A long-lived development branch to build an experimental CPU backend for [Triton](https://github.com/triton-lang/triton).

This repository clones the main Triton repository, but we intend to minimize
divergences in the core (and ideally upstream anything that needs to change and
isn't too CPU-specific).  Most of the CPU work should be in a backend
subdirectory (similar to how GPU vendors are supported today).  We're starting
with a clone to give ourselves maximum development flexibility as this project
gets off the ground!

# Getting started

The recommended way to set up a development environment is via [`uv`](https://docs.astral.sh/uv/):

```shell
git submodule update --init
uv venv --python 3.12 ; source .venv/bin/activate
uv pip install -r python/requirements.txt \
               -r python/test-requirements.txt \
               -r python/tutorials/requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . --no-build-isolation
```

See below in the upstream README for further instructions, e.g. to
[build with a custom LLVM](#building-with-a-custom-llvm).

Smoke-test your environment by running the following samples:

```shell
uv run python python/tutorials/cpu/01-vector-add.py
uv run python python/tutorials/cpu/08-sfc-matmul.py
```

The samples use `triton.runtime.driver.set_active_to_cpu()` to switch to CPU
usage. Alternatively, set `TRITON_DEFAULT_BACKEND=cpu` to select the CPU
backend explicitly. When no GPU backend is active, Triton selects CPU
automatically.

**NOTE: It's still work in progress.**

---

# Upstream README

| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

# Triton Conference 2025

![Triton Banner](https://github.com/user-attachments/assets/b4b6972a-857c-417f-bf2c-f16f38a358c0)

The 3rd Triton Developer Conference took place on October 21, 2025 at the Microsoft Silicon Valley Campus in Mountain View, California.

### Conference Materials

Conference recordings and materials are now available online:

- **Conference Videos:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO)
- **Conference Slides:** [Google Drive Folder](https://drive.google.com/drive/folders/1KB6tD3UM1J0_eUp-F-JSlGrargLBawIr)

For previous conference materials, see:
- [2024 Conference Materials](docs/meetups/dev_conference_2024.md)
- [2023 Conference Materials](docs/meetups/dev-meetup-2023.md)

# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.  See also these third-party [Triton puzzles](https://github.com/srush/Triton-Puzzles), which can all be run using the Triton interpreter -- no GPU required.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.10-3.14.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build and use LLVM from source.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

For convenience, use the following command to build LLVM and install Triton with the custom LLVM:

```shell
make dev-install-llvm
```

<details>
<summary>
Alternatively, follow these steps to build LLVM from source manually.
</summary>

1. Find the version of LLVM that Triton builds against. Check the `llvm_hash`
field in `cmake/llvm-info.json` to see the current version. For example, if it
says:
       "llvm_hash": "49af6502c6dcb4a7f7520178bd14df396f78240c"

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run:

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables:

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e .

</details>

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e .` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- The build system creates a `compile_commands.json` file under the Triton repo
  directory. This file is used by VSCode IntelliSense and clangd to provide
  code completion and other features for C++ code.

  If IntelliSense does not work, you can try the following steps:

    - Do a local build. Run command `pip install -e .`.
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find ./build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`.
    - In VSCode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe:

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

