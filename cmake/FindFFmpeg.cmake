cmake_minimum_required(VERSION 3.16)

# FindFFmpeg.cmake
# Locates the FFmpeg libraries required for video encoding.
#
# Creates imported targets:
#   FFmpeg::avcodec
#   FFmpeg::avformat
#   FFmpeg::avutil
#   FFmpeg::swscale
#
# Sets FFmpeg_FOUND to TRUE when all four are found.

find_package(PkgConfig QUIET)

function(_ffmpeg_find_component comp header)
    if (TARGET FFmpeg::${comp})
        return()
    endif ()

    # Try pkg-config first (Linux / Mac / MSYS2)
    if (PkgConfig_FOUND)
        pkg_check_modules(PC_FFmpeg_${comp} QUIET "lib${comp}")
    endif ()

    find_path(FFmpeg_${comp}_INCLUDE_DIR
        NAMES "lib${comp}/${header}"
        HINTS
            ${PC_FFmpeg_${comp}_INCLUDEDIR}
            ${PC_FFmpeg_${comp}_INCLUDE_DIRS}
        PATH_SUFFIXES ffmpeg
    )
    mark_as_advanced(FFmpeg_${comp}_INCLUDE_DIR)

    find_library(FFmpeg_${comp}_LIBRARY
        NAMES ${comp}
        HINTS
            ${PC_FFmpeg_${comp}_LIBDIR}
            ${PC_FFmpeg_${comp}_LIBRARY_DIRS}
    )
    mark_as_advanced(FFmpeg_${comp}_LIBRARY)

    if (FFmpeg_${comp}_INCLUDE_DIR AND FFmpeg_${comp}_LIBRARY)
        add_library(FFmpeg::${comp} UNKNOWN IMPORTED)
        set_target_properties(FFmpeg::${comp} PROPERTIES
            IMPORTED_LOCATION "${FFmpeg_${comp}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFmpeg_${comp}_INCLUDE_DIR}"
        )
        if (PC_FFmpeg_${comp}_FOUND)
            set_property(TARGET FFmpeg::${comp} APPEND PROPERTY
                INTERFACE_COMPILE_OPTIONS "${PC_FFmpeg_${comp}_CFLAGS_OTHER}")
        endif ()
    endif ()
endfunction()

_ffmpeg_find_component(avcodec avcodec.h)
_ffmpeg_find_component(avformat avformat.h)
_ffmpeg_find_component(avutil avutil.h)
_ffmpeg_find_component(swscale swscale.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS
        FFmpeg_avcodec_INCLUDE_DIR FFmpeg_avcodec_LIBRARY
        FFmpeg_avformat_INCLUDE_DIR FFmpeg_avformat_LIBRARY
        FFmpeg_avutil_INCLUDE_DIR  FFmpeg_avutil_LIBRARY
        FFmpeg_swscale_INCLUDE_DIR FFmpeg_swscale_LIBRARY
)
