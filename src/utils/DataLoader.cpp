//
// Created by fabian on 16/08/2026.
//

#include "../../include/utils/DataLoader.h"
#include <iostream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/lib/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../include/lib/stb_image_resize2.h"

Matrix3D DataLoader::load_image(const std::string& filepath, int target_width, int target_height) {
	int original_width, original_height, channels;

	unsigned char* img_data = stbi_load(filepath.c_str(), &original_width, &original_height, &channels, 3);

	if (!img_data) {
		throw std::runtime_error("Failed to load image: " + filepath);
	}

	auto* resized_data = new unsigned char[target_width * target_height * 3];

	stbir_resize_uint8_linear(img_data, original_width, original_height, 0,
						  resized_data, target_width, target_height, 0, STBIR_RGB);

	stbi_image_free(img_data);

	Matrix3D tensor(3, target_height, target_width);

	// Loop through every pixel in the resized image
	for (int y = 0; y < target_height; ++y) {
		for (int x = 0; x < target_width; ++x) {
			for (int c = 0; c < 3; ++c) {
				// Calculate the 1D index of the interleaved RGB array
				const int index = (y * target_width + x) * 3 + c;

				// Extract the value (0-255), normalize it to (0.0-1.0), and
				// assign it to the specific Depth (c), Row (y), and Column (x)
				tensor(c, y, x) = resized_data[index] / 255.0f;
			}
		}
	}

	// Clean up the resized array
	delete[] resized_data;

	return tensor;
}