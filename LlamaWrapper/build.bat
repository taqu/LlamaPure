cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="install"
cmake --build build --config Release
cmake --install build
xcopy /Y .\install\bin\*.dll ..\LlamaPure\runtimes\win-x64\native