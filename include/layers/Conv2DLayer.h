//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_CONV2DLAYER_H
#define IMAGE_CLASSIFIER_CONV2DLAYER_H

#include <vector>
#include <cstddef>
#include "../utils/Matrix3D.h"

class Conv2DLayer {
private:
	size_t num_filters_;
	size_t filter_size_;
	size_t input_depth_;

	std::vector<Matrix3D> filters_;

	std::vector<float> biases_;

public:
	Conv2DLayer(size_t num_filters, size_t filter_size, size_t input_depth);

	[[nodiscard]] Matrix3D forward_pass(const Matrix3D& input) const;

	void print_filter(size_t filter_index) const;
};



#endif //IMAGE_CLASSIFIER_CONV2DLAYER_H
