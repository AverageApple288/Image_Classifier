//
// Created by fabian on 16/08/2026.
//

#include "../../include/layers/Conv2DLayer.h"
#include "../../include/utils/Matrix3D.h"

#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Constructor Implementation
Conv2DLayer::Conv2DLayer(const size_t num_filters, const size_t filter_size, const size_t input_depth)
    : num_filters_(num_filters), filter_size_(filter_size), input_depth_(input_depth)
{
    // Calculate the number of input connections to a single filter
    const auto n_in = static_cast<float>(input_depth_ * filter_size_ * filter_size_);

    // Calculate the standard deviation for He Initialization (for ReLU)
    const float std_dev = std::sqrt(2.0 / n_in);

    // Set up the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // Define the normal distribution centered at 0.0 with the calculated spread
    std::normal_distribution<float> dist(0.0, std_dev);

    // Generate the filters
    for (size_t i = 0; i < num_filters_; ++i) {
        // Create a new empty filter Matrix3D
        Matrix3D filter(input_depth_, filter_size_, filter_size_, 0.0);

        // Populate it with random numbers from our distribution
        for (size_t d = 0; d < input_depth_; ++d) {
            for (size_t r = 0; r < filter_size_; ++r) {
                for (size_t c = 0; c < filter_size_; ++c) {
                    filter(d, r, c) = dist(gen);
                }
            }
        }

        // Add the fully initialized filter to our layer's collection
        filters_.push_back(filter);

        // Initialize biases to 0.0
        biases_.push_back(0.0f);
    }
}

// Convolution logic
Matrix3D Conv2DLayer::forward_pass(const Matrix3D &input) const {
	const size_t out_rows = input.rows() - filter_size_ + 1;
	const size_t out_cols = input.cols() - filter_size_ + 1;

	Matrix3D output(num_filters_, out_rows, out_cols);

	for (size_t f = 0; f < num_filters_; ++f) {
		const auto& current_filter = filters_[f];
		const float current_bias = biases_[f];

		for (size_t y = 0 ; y < out_rows; ++y) {
			for (size_t x = 0; x < out_cols; ++x) {
				float sum = 0.0;

				for (size_t d = 0; d < input_depth_; ++d) {
					for (size_t r = 0; r < filter_size_; ++r) {
						for (size_t c = 0; c < filter_size_; ++c) {
							sum += input(d, y + r, x + c) * current_filter(d, r, c);
						}
					}
				}

				output(f, y, x) = sum + current_bias;
			}
		}
	}

	return output;
}

// Debugging Method
void Conv2DLayer::print_filter(const size_t filter_index) const {
    if (filter_index >= num_filters_) {
        throw std::out_of_range("Filter index out of bounds.");
    }

    std::cout << "--- Filter " << filter_index << " Weights ---\n";
    filters_[filter_index].print();
}