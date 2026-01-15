//
// Created by fabian on 13/01/2026.
//

#include "../include/ImageProcessor.h"
#define STB_IMAGE_IMPLEMENTATION // Essential: Turns the header into code
#include "../include/stb_image.h"
#include <iostream>
#include <cmath>
#include <cmath>

Matrix ImageProcessor::load_and_process(const std::string& filepath) {
    int w, h, channels;

    // 1. DECODE: Load raw bytes
    // force_channels=1 loads it as Grayscale automatically
    unsigned char* img_data = stbi_load(filepath.c_str(), &w, &h, &channels, 1);

    if (img_data == nullptr) {
        std::cerr << "Failed to load image: " << filepath << std::endl;
        return Matrix(target_height, target_width); // Return empty on fail
    }

    // 2. CONVERT: Raw bytes -> Matrix
    Matrix raw_matrix(h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            // Access linear array: index = (row * width) + col
            unsigned char pixel_val = img_data[i * w + j];

            // Normalize (0-255 -> 0.0-1.0)
            raw_matrix.at(i, j) = static_cast<float>(pixel_val) / 255.0f;
        }
    }

    // Free the memory allocated by stb_image
    stbi_image_free(img_data);

    // 3. RESIZE: Run our manual resizing algorithm
    if (w != target_width || h != target_height) {
        return resize(raw_matrix);
    }

    return raw_matrix;
}

// Manual "Nearest Neighbor" Resizing Algorithm
Matrix ImageProcessor::resize(const Matrix& input) {
    Matrix scaled(target_height, target_width);

    // Calculate scaling factors
    // e.g., if input is 100x100 and target is 10x10, ratio is 10.0
    float x_ratio = (float)input.cols / target_width;
    float y_ratio = (float)input.rows / target_height;

    for (int i = 0; i < target_height; ++i) {
        for (int j = 0; j < target_width; ++j) {

            // Find the corresponding coordinate in the source image
            // "floor" snaps to the nearest whole pixel
            int src_y = (int)floor(i * y_ratio);
            int src_x = (int) std::floor(j * x_ratio);

            // Safety check to stay in bounds
            if (src_y >= input.rows) src_y = input.rows - 1;
            if (src_x >= input.cols) src_x = input.cols - 1;

            scaled.at(i, j) = input.at(src_y, src_x);
        }
    }

    return scaled;
}