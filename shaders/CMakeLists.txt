compile_shaders(
    TARGET shaders
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/triangle.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/triangle.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/shadow.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/shadow.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/skybox.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/skybox.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/volumetric.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/volumetric.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/fullscreen.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/bloom_extract.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/bloom_blur.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/bloom_composite.frag"
)
