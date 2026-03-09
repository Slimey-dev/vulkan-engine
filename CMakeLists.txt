cmake_minimum_required(VERSION 3.24)
project(VulkanEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan 1.3 REQUIRED COMPONENTS glslc)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompileShaders)

add_subdirectory(engine)
add_subdirectory(shaders)
add_subdirectory(app)
