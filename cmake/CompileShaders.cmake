function(compile_shaders)
    cmake_parse_arguments(SHADER "" "TARGET" "SOURCES" ${ARGN})

    set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

    set(SPV_OUTPUTS "")

    foreach(SOURCE ${SHADER_SOURCES})
        get_filename_component(FILENAME "${SOURCE}" NAME)
        set(OUTPUT "${SHADER_OUTPUT_DIR}/${FILENAME}.spv")

        add_custom_command(
            OUTPUT "${OUTPUT}"
            COMMAND Vulkan::glslc "${SOURCE}" -o "${OUTPUT}"
            DEPENDS "${SOURCE}"
            COMMENT "Compiling shader ${FILENAME} -> ${FILENAME}.spv"
            VERBATIM
        )

        list(APPEND SPV_OUTPUTS "${OUTPUT}")
    endforeach()

    add_custom_target(${SHADER_TARGET} DEPENDS ${SPV_OUTPUTS})
endfunction()
