cmake_minimum_required(VERSION 3.30)
project(SokolCSTest)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd_party)

add_executable(GLparticle particle_gl.cpp)
target_link_libraries(GLparticle PRIVATE sokol HandmadeMath)

add_executable(DXparticle particle_dx.cpp)
target_link_libraries(DXparticle PRIVATE sokol HandmadeMath)
