//
// Created by fabian on 16/08/2026.
//

#include "../../include/layers/PoolingLayer.h"

PoolingLayer::PoolingLayer(const size_t window_size, const size_t stride, const size_t input_depth) : window_size_(window_size), stride_(stride), input_depth_(input_depth) {
}

PoolResult PoolingLayer::forward_pass(const Matrix3D &input) const {
	const size_t out_rows = (input.rows() - window_size_) / stride_ + 1;
	const size_t out_cols = (input.cols() - window_size_) / stride_ + 1;

	Matrix3D output(input_depth_, out_rows, out_cols);
	Matrix3D argmax_map(input_depth_, input.rows(), input.cols());

	for (size_t d = 0; d < input_depth_; ++d) {
		for (size_t out_y = 0; out_y < out_rows; ++out_y) {
			for (size_t out_x = 0; out_x < out_cols; ++out_x) {
				float max_value = -std::numeric_limits<float>::infinity();
				size_t win_y = 0;
				size_t win_x = 0;

				for (size_t r = 0; r < window_size_; ++r) {
					for (size_t c = 0; c < window_size_; ++c) {
						const size_t in_y = out_y * stride_ + r;
						const size_t in_x = out_x * stride_ + c;

						if (input(d, in_y, in_x) > max_value) {
							max_value = input(d, in_y , in_x);
							win_y = in_y;
							win_x = in_x;
						}
					}
				}

				output(d, out_y, out_x) = max_value;
				argmax_map(d, win_y, win_x) = 1.0f;  // Mark the position of the max value
			}
		}
	}

	return PoolResult{output, argmax_map};
}

Matrix3D PoolingLayer::backward_pass(const Matrix3D& d_output, const Matrix3D& argmax_map) const {
	Matrix3D d_input(argmax_map.depth(), argmax_map.rows(), argmax_map.cols(), 0.0f);

	const size_t out_rows = d_output.rows();
	const size_t out_cols = d_output.cols();

	for (size_t d = 0; d < input_depth_; ++d) {
		for (size_t out_y = 0; out_y < out_rows; ++out_y) {
			for (size_t out_x = 0; out_x < out_cols; ++out_x) {
				// The gradient trying to flow backward into this 2x2 window
				const float grad = d_output(d, out_y, out_x);

				// Find which of the 4 pixels gets the gradient
				for (size_t r = 0; r < window_size_; ++r) {
					for (size_t c = 0; c < window_size_; ++c) {
						const size_t in_y = out_y * stride_ + r;
						const size_t in_x = out_x * stride_ + c;

						// If the forward pass marked this as the max value, give it the gradient
						if (argmax_map(d, in_y, in_x) == 1.0f) {
							d_input(d, in_y, in_x) += grad;
						}
					}
				}
			}
		}
	}

	return d_input;
}
