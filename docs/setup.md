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