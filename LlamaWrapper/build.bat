cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="install"
REM -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" .
cmake --build build --config Release
cmake --install build --config Release
xcopy /Y .\install\bin\*.* ..\LlamaPure\runtimes\win-x64\native