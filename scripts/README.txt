Place this entire scripts folder inside your project root:

Candle\
├── CMakeLists.txt
├── .venv\
├── bindings.cpp
├── kernel.cu
├── kernel.h
├── test.py
└── scripts\
    ├── setup.bat
    ├── configure.bat
    ├── build.bat
    ├── clean.bat
    ├── rebuild.bat
    ├── test.bat
    └── all.bat

Run from the project root:

    .\scripts\setup.bat
    .\scripts\configure.bat
    .\scripts\build.bat
    .\scripts\test.bat

For a clean rebuild:

    .\scripts\rebuild.bat

To run everything:

    .\scripts\all.bat
