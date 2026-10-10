Run the scripts from the project root on a Windows machine with CUDA and
Visual Studio Build Tools installed:

    .\scripts\setup.bat
    .\scripts\configure.bat
    .\scripts\build.bat
    .\scripts\test.bat

For a clean rebuild:

    .\scripts\rebuild.bat

To run setup, rebuild, and tests:

    .\scripts\all.bat

Project layout:

    Candle\
    |-- CMakeLists.txt
    |-- cpp\
    |   |-- bindings.cpp
    |   |-- tensor.cpp
    |   |-- tensor.h
    |   |-- utils.cpp
    |   |-- utils.h
    |   `-- kernels\
    |-- python\
    |   |-- candle.py
    |   |-- test_candle.py
    |   `-- numeric_test_candle.py
    `-- scripts\

GEMM animations (CPU-only; no CUDA build or profiler needed):

    .\.venv\Scripts\python.exe -m pip install Pillow
    .\.venv\Scripts\python.exe scripts\visualize_gemm_naive.py
    .\.venv\Scripts\python.exe scripts\visualize_gemm_smem.py
    .\.venv\Scripts\python.exe scripts\visualize_gemm_2d_tile.py
    .\.venv\Scripts\python.exe scripts\visualize_gemm_vectorized.py

Each script writes its own looping GIF to scripts\gifs\, regardless of the
working directory. All four share a drawing helper, gemm_visualization.py.
Use --output path.gif to choose another location, --width 960 for a smaller
image, --fps 15 for smoother movement, and --preview for a PNG contact sheet.

The diagrams show a simple load -> compute -> store flow. A is teal, B is
purple, and C/accumulators are green. Whole rows, columns, or tiles move in one
animation, aggregating the scalar loop iterations so the GIF can accompany
a written explanation. No numeric counters or instruction-by-instruction
animation is shown. Playback does not compare kernel performance.

Naive: a whole A row and B column move directly to the thread.
Shared memory: whole input tiles move to shared memory, then one output is
computed per thread. Two K tiles are shown.
2D tiled: the same shared-memory path feeds a 2x2 register/output tile.
Vectorized: four adjacent values travel in outlined float4 groups. A's groups
turn into columns in shared memory; B's groups stay in rows. A 4x4 register
tile is written back in groups of four.

The smaller diagram tiles are for readability. Actual CUDA configurations:
    naive:      16x16 threads/block, 1 output/thread
    smem:       16x16 shared tiles, 1 output/thread
    2d_tile:    BM=BN=64, BK=8, TM=TN=4
    vectorized: BM=BN=128, BK=8, TM=TN=8
