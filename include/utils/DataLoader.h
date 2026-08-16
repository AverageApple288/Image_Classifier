//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_DATALOADER_H
#define IMAGE_CLASSIFIER_DATALOADER_H

#include "Matrix3D.h"
#include <string>

class DataLoader {
public:
	// Loads an image, resizes it, and converts it to a normalized Matrix3D tensor
	static Matrix3D load_image(const std::string& filepath, int target_width, int target_height);
};


#endif //IMAGE_CLASSIFIER_DATALOADER_H
