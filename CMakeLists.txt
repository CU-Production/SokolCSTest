cmake_minimum_required(VERSION 3.30)
project(SokolCSTest)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd_party)

add_executable(GLparticle main_gl.cpp)
target_link_libraries(GLparticle PRIVATE sokol HandmadeMath)
