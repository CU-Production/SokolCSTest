cmake_minimum_required(VERSION 3.30)
project(SokolCSTest)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd_party)

add_executable(GLparticle particle_gl.cpp)
target_link_libraries(GLparticle PRIVATE sokol HandmadeMath)

add_executable(DXparticle particle_dx.cpp)
target_link_libraries(DXparticle PRIVATE sokol HandmadeMath)

add_executable(GLnoise noise_gl.cpp)
target_link_libraries(GLnoise PRIVATE sokol HandmadeMath)

add_executable(DXnoise noise_dx.cpp)
target_link_libraries(DXnoise PRIVATE sokol HandmadeMath)

add_executable(GLraymarching raymarching_gl.cpp)
target_link_libraries(GLraymarching PRIVATE sokol HandmadeMath)
add_custom_command(TARGET GLraymarching POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:GLraymarching>/" "${CMAKE_CURRENT_SOURCE_DIR}/raymarching_gl.glsl"
    COMMAND_EXPAND_LISTS
)

add_executable(DXraymarching raymarching_dx.cpp)
target_link_libraries(DXraymarching PRIVATE sokol HandmadeMath)
add_custom_command(TARGET DXraymarching POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:DXraymarching>/" "${CMAKE_CURRENT_SOURCE_DIR}/raymarching_dx.hlsl"
    COMMAND_EXPAND_LISTS
)
