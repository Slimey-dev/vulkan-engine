# FindSteamAudio.cmake — locate vendored Steam Audio SDK

set(STEAMAUDIO_ROOT "${CMAKE_SOURCE_DIR}/third_party/steamaudio")

find_path(STEAMAUDIO_INCLUDE_DIR phonon.h PATHS "${STEAMAUDIO_ROOT}/include" NO_DEFAULT_PATH)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(STEAMAUDIO_LIBRARY phonon PATHS "${STEAMAUDIO_ROOT}/lib/linux-x86_64" NO_DEFAULT_PATH)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_library(STEAMAUDIO_LIBRARY phonon PATHS "${STEAMAUDIO_ROOT}/lib/macos" NO_DEFAULT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SteamAudio DEFAULT_MSG STEAMAUDIO_LIBRARY STEAMAUDIO_INCLUDE_DIR)

if(SteamAudio_FOUND AND NOT TARGET SteamAudio::phonon)
    add_library(SteamAudio::phonon SHARED IMPORTED)
    set_target_properties(SteamAudio::phonon PROPERTIES
        IMPORTED_LOCATION "${STEAMAUDIO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${STEAMAUDIO_INCLUDE_DIR}"
    )
endif()
