compile_shaders(
    TARGET shaders
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/triangle.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/triangle.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/shadow.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/shadow.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/skybox.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/skybox.frag"
)
