//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_MATRIX_H
#define IMAGE_CLASSIFIER_MATRIX_H

#include <vector>
#include <iostream>

class Matrix3D {
private:
	size_t depth_;
	size_t rows_;
	size_t cols_;
	std::vector<double> data_;

public:
	// Constructor
	Matrix3D(size_t depth, size_t rows, size_t cols, double initial_value = 0.0);

	// Getters
	[[nodiscard]] size_t depth() const;
	[[nodiscard]] size_t rows() const;
	[[nodiscard]] size_t cols() const;

	// Index operators
	double& operator()(size_t d, size_t r, size_t c);
	const double& operator()(size_t d, size_t r, size_t c) const;

	// Utility
	void print() const;
};



#endif //IMAGE_CLASSIFIER_MATRIX_H
