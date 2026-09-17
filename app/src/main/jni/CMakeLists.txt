project(YOLOWORLD)

cmake_minimum_required(VERSION 3.10.2)

# NOTE: this demo deliberately does NOT link opencv-mobile. The prebuilt
# opencv-mobile 2.4.13.7 static libs in this SDK were built with a newer NDK and
# reference std::__ndk1::__libcpp_verbose_abort() / __kmpc_dispatch_deinit(),
# which do not exist in NDK 21.3. All image handling now goes through
# ywmimage.h (plain malloc'd buffer + hand-rolled box/text drawing).

# ncnn (android-vulkan build) — adjust if your SDK lives elsewhere
set(ncnn_DIR E:/AndroidSDK/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnn_yoloworld SHARED yoloworldncnn.cpp yoloworld.cpp ndkcamera.cpp)

target_link_libraries(ncnn_yoloworld ncnn camera2ndk mediandk)
