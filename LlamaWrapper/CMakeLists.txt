cmake_minimum_required(VERSION 3.50)
project(LlamaWrapper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Build app" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build tools" FORCE)

add_subdirectory(llama.cpp)
