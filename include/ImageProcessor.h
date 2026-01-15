//
// Created by fabian on 13/01/2026.
//

#ifndef IMAGE_CLASSIFIER_IMAGE_H
#define IMAGE_CLASSIFIER_IMAGE_H
#include <string>

#include "Matrix.h"


class ImageProcessor {
public:
	// Target dimensions for our model (e.g., 32x32)
	int target_width;
	int target_height;

	ImageProcessor(int w, int h) : target_width(w), target_height(h) {}

	// The main function: Path -> Ready-to-use Matrix
	Matrix load_and_process(const std::string& filepath);

private:
	// Helper: Manual resizing algorithm (Nearest Neighbor)
	Matrix resize(const Matrix& input);
};


#endif //IMAGE_CLASSIFIER_IMAGE_H