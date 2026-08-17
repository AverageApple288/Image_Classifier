//
// Created by fabian on 17/08/2026.
//

#ifndef IMAGE_CLASSIFIER_MODEL_H
#define IMAGE_CLASSIFIER_MODEL_H
#include <string>
#include <vector>
#include "../utils/Matrix3D.h"

class Model {
public:
	Model();

	void add_image(const std::string& filepath);

private:
	std::vector<Matrix3D> output_feature_maps;
};



#endif //IMAGE_CLASSIFIER_MODEL_H
