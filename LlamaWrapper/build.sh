#!/usr/bin/bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="install"
cmake --build build --config Release
cmake --install build --config Release
cp -f ./install/bin/*.* ../LlamaPure/runtimes/linux-x64/native
