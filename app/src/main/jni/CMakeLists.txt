project(YOLOWORLD)

cmake_minimum_required(VERSION 3.10.2)

# NOTE: this demo deliberately does NOT link opencv-mobile. The prebuilt
# opencv-mobile 2.4.13.7 static libs in this SDK were built with a newer NDK and
# reference std::__ndk1::__libcpp_verbose_abort() / __kmpc_dispatch_deinit(),
# which do not exist in NDK 21.3. All image handling now goes through
# ywmimage.h (plain malloc'd buffer + hand-rolled box/text drawing).

# ncnn 20260526 CPU-only (android build) — 正式 YOLO-World 模型需要新版算子
# (Einsum/Normalize/Gemm/MemoryData)，20210525 老库不支持。
# 不用 vulkan 包：其携带的 glslang 与 NDK 21.3 的 libc++ ABI 不兼容。
set(ncnn_DIR E:/AndroidSDK/ncnn-20260526-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnn_yoloworld SHARED yoloworldncnn.cpp yoloworld.cpp ndkcamera.cpp)

target_link_libraries(ncnn_yoloworld ncnn camera2ndk mediandk)
