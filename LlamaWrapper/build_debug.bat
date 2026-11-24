cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="install"
REM -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" .
cmake --build build --config Debug
cmake --install build --config Debug
xcopy /Y .\install\bin\*.* ..\LlamaPure\runtimes\win-x64\native