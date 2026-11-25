# Setup

## Python environment
- Python 3.13 venv in `.venv`
- `.venv/bin/python -m pip install -r requirements.txt`

## SimGrid 4.1 (macOS, Apple Silicon)
No macOS wheel: pip builds it from source (about N minutes).

1. `brew install boost` (do not install Homebrew's pybind11: 3.x breaks SimGrid 4.1's bindings)
2. `.venv/bin/python -m pip install setuptools wheel "pybind11<3"`
3. `pybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)" .venv/bin/python -m pip install -v --no-build-isolation simgrid`
4. The extension has no rpath, so it cannot find libsimgrid next to it. After every install:
   - `install_name_tool -add_rpath @loader_path .venv/lib/python3.13/site-packages/simgrid.cpython-313-darwin.so`
   - `codesign --force --sign - .venv/lib/python3.13/site-packages/simgrid.cpython-313-darwin.so`
5. Check: `.venv/bin/python spikes/simgrid_ptask.py` prints 5 lines ending in `ok`

Versions that worked: SimGrid 4.1, Boost 1.92.0, pybind11 2.x, cmake 4.3.1, Apple clang 17.

## SCALE-Sim 3.0.0
- `pip install scalesim` (pure Python), but it needs `numpy<2.4`: newer NumPy makes
  `int(max(array))` in double_buffered_scratchpad_mem.py an error
- GEMM topology rows are `name, M, N, K,` (A is M x K, B is K x N); every row must end with a comma
- a layout file is always opened: pass configs/scalesim/no_layout.csv
- word size is hard-coded to 1 byte: sram "kB" in the cfg = thousands of elements,
  dram bytes = (DRAM IFMAP Reads + DRAM Filter Reads + DRAM OFMAP Writes) x dtype_bytes
- InterfaceBandwidth CALC => 0 stall cycles, Total Cycles = compute only
- dataflow os: 20351 cycles, 1.34 MB for a 256^3 bf16 gemm (ws and is: 22399 cycles, 2.23 MB)
- check: `.venv/bin/python spikes/scalesim_gemm.py`

## Ramulator 2.1 (commit 7bead6c, cloned into tools/ramulator2, not committed)
1. `git clone https://github.com/CMU-SAFARI/ramulator2.git tools/ramulator2` (run from the repo root)
2. Apple clang 17 patch: `src/ramulator/base/param.h:37` `.as<T>()` -> `.template as<T>()`
3. `.venv/bin/python -m pip install pyyaml`
4. `cmake -S tools/ramulator2 -B tools/ramulator2/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python" -DCMAKE_POLICY_VERSION_MINIMUM=3.5`
5. `cmake --build tools/ramulator2/build`
6. `.venv/bin/python -m pip install -e tools/ramulator2`
7. If `import ramulator` fails: `chflags -R nohidden .venv` (Python 3.13 skips hidden .pth files)
8. Check: `.venv/bin/python spikes/ramulator_streams.py` gives fractions 0.86 0.84 0.77 0.68 0.55 0.40

Finding: the loss curve's shape is set by the address mapping. MOP4CLXOR (xor bank hashing) is the
realistic one; RoBaRaCoCh with 1 stream reaches only 75% of peak, and aligned streams collapse onto one bank.


## gem5 25.1.0.1 (in Docker, cloned into tools/gem5, not committed)
gem5 builds and runs only on Linux, so it runs in the official gem5 build image on an arm64 VM.

1. Docker VM: `brew install colima docker`, then
   `colima start --cpu 8 --memory 12 --disk 50 --vm-type vz --mount-type virtiofs`
   - check: `docker run --rm ubuntu:24.04 uname -m` prints `aarch64` (native, no emulation)
   - leftover Docker Desktop config gives "docker-credential-desktop not found": remove the line with
     `sed -i '' '/"credsStore"/d' ~/.docker/config.json`
2. `git clone --depth 1 --branch v25.1.0.1 https://github.com/gem5/gem5.git tools/gem5` (commit c8222cc)
3. `docker pull ghcr.io/gem5/ubuntu-24.04_all-dependencies:v25-1`
4. Build (about 2-3 h; keep the Mac awake with `caffeinate -i`, no internet needed once the image is pulled):
   `docker run --rm -it -u "$(id -u):$(id -g)" -v "$PWD/tools/gem5":/gem5 -w /gem5 ghcr.io/gem5/ubuntu-24.04_all-dependencies:v25-1 scons build/ARM/gem5.opt -j 2`
   - `-j 4` ran out of memory in the 12 GB VM ("Killed signal terminated program cc1plus"); rerunning
     with a lower `-j` continues where it stopped
5. Check: `docker run --rm -v "$PWD/tools/gem5":/gem5 -w /gem5 ghcr.io/gem5/ubuntu-24.04_all-dependencies:v25-1 build/ARM/gem5.opt -B`
   prints `gem5 version 25.1.0.1` (gem5 has no `--version` flag)

The default ARM build uses Ruby with the CHI protocol; SE-mode runs can still use the classic caches.