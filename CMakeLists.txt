cmake_minimum_required(VERSION 3.30)
project(SokolCSTest)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd_party)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE sokol)
