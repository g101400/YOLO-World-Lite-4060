// Minimal, dependency-free image container.
//
// This replaces the small subset of OpenCV this demo used to rely on:
//   cv::Mat / CV_8UC1 / CV_8UC3   ->  YWMat
//   cv::Rect_<float> / operator&  ->  YWRect
//
// Why: the prebuilt opencv-mobile 2.4.13.7 static libs in this SDK were built
// with a newer NDK than the one used here (NDK 21.3). Their objects reference
// std::__ndk1::__libcpp_verbose_abort() and __kmpc_dispatch_deinit(), neither of
// which exists in NDK 21.3's libc++/libomp. Rather than stub OpenMP internals,
// we drop the dependency: the demo only needs a buffer holder.
//
// NOTE: no pixel drawing lives here anymore. Boxes / labels / status messages are
// rendered by the Android overlay view (YoloOverlayView), which can use the
// system font and therefore supports Chinese prompts. The native side only runs
// inference and reports boxes.

#ifndef YWMIMAGE_H
#define YWMIMAGE_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- YWRect ----

struct YWRect
{
    float x;
    float y;
    float width;
    float height;

    YWRect() : x(0.f), y(0.f), width(0.f), height(0.f) {}
    YWRect(float _x, float _y, float _w, float _h) : x(_x), y(_y), width(_w), height(_h) {}

    float area() const
    {
        if (width <= 0.f || height <= 0.f) return 0.f;
        return width * height;
    }
};

inline YWRect operator&(const YWRect& a, const YWRect& b)
{
    float x1 = a.x > b.x ? a.x : b.x;
    float y1 = a.y > b.y ? a.y : b.y;
    float ax2 = a.x + a.width;
    float bx2 = b.x + b.width;
    float ay2 = a.y + a.height;
    float by2 = b.y + b.height;
    float x2 = ax2 < bx2 ? ax2 : bx2;
    float y2 = ay2 < by2 ? ay2 : by2;
    if (x2 <= x1 || y2 <= y1) return YWRect(0.f, 0.f, 0.f, 0.f);
    return YWRect(x1, y1, x2 - x1, y2 - y1);
}

// ----------------------------------------------------------------- YWMat ----

// Interleaved pixel buffer: 1 channel (yuv) or 3 channels (rgb).
class YWMat
{
public:
    unsigned char* data;
    int rows;
    int cols;
    int channels;

    YWMat() : data(0), rows(0), cols(0), channels(0) {}

    YWMat(int _rows, int _cols, int _channels)
        : data(0), rows(_rows), cols(_cols), channels(_channels)
    {
        size_t n = (size_t)rows * cols * channels;
        if (n) data = (unsigned char*)malloc(n);
    }

    YWMat(const YWMat& o) : data(0), rows(o.rows), cols(o.cols), channels(o.channels)
    {
        size_t n = (size_t)rows * cols * channels;
        if (n)
        {
            data = (unsigned char*)malloc(n);
            memcpy(data, o.data, n);
        }
    }

    YWMat& operator=(const YWMat& o)
    {
        if (this == &o) return *this;
        free(data);
        data = 0;
        rows = o.rows;
        cols = o.cols;
        channels = o.channels;
        size_t n = (size_t)rows * cols * channels;
        if (n)
        {
            data = (unsigned char*)malloc(n);
            memcpy(data, o.data, n);
        }
        return *this;
    }

    ~YWMat() { free(data); }

    bool empty() const { return data == 0; }

    unsigned char* ptr(int y) { return data + (size_t)y * cols * channels; }
    const unsigned char* ptr(int y) const { return data + (size_t)y * cols * channels; }
};

#endif // YWMIMAGE_H
