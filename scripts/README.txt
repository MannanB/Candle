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
