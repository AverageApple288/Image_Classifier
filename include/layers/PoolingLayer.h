//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_POOLINGLAYER_H
#define IMAGE_CLASSIFIER_POOLINGLAYER_H

#include <cstddef>
#include "../utils/Matrix3D.h"

struct PoolResult {
	Matrix3D output;
	Matrix3D mask;
};

class PoolingLayer {
public:
	PoolingLayer(size_t window_size, size_t stride, size_t input_depth);

	[[nodiscard]] PoolResult forward_pass(const Matrix3D &input) const;

	[[nodiscard]] Matrix3D backward_pass(const Matrix3D& d_output, const Matrix3D& argmax_map) const;

private:
	size_t window_size_;
	size_t stride_;
	size_t input_depth_;
};



#endif //IMAGE_CLASSIFIER_POOLINGLAYER_H
