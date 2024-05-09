cmake_minimum_required(VERSION 3.0)
project(autograd.cpp)

set(CMAKE_CXX_STANDARD 11)
set(HOME ./)
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin)

# used by clangd to generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

message("current directtory:"${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${MAIN_HEAD})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(app ${MAIN_SRC})