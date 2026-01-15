//
// Created by fabian on 13/01/2026.
//

#ifndef IMAGE_CLASSIFIER_MATRIX_H
#define IMAGE_CLASSIFIER_MATRIX_H
#include <vector>


class Matrix {
public:
	std::vector<float> data;
	int rows, cols;

	Matrix(int r, int c);

	float &at(int r, int c);

	const float &at(int r, int c) const;

	static Matrix matmul(const Matrix& A, const Matrix& B);
};


#endif //IMAGE_CLASSIFIER_MATRIX_H