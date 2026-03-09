add_executable(triangle main.cpp)
target_link_libraries(triangle PRIVATE VulkanEngine)
add_dependencies(triangle shaders)
