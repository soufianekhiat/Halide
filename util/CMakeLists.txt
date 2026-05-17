find_package(FFmpeg REQUIRED)

add_executable(HalideTraceViz HalideTraceViz.cpp)
target_link_libraries(HalideTraceViz PRIVATE
    Halide::Halide Halide::Tools
    FFmpeg::avcodec FFmpeg::avformat FFmpeg::avutil FFmpeg::swscale)

add_executable(HalideTraceDump HalideTraceDump.cpp HalideTraceUtils.cpp)
target_link_libraries(HalideTraceDump PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)
