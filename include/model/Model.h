//
// Created by fabian on 17/08/2026.
//

#ifndef IMAGE_CLASSIFIER_MODEL_H
#define IMAGE_CLASSIFIER_MODEL_H
#include <string>
#include <vector>

#include "../layers/Conv2DLayer.h"
#include "../utils/Matrix3D.h"

class Model {
public:
	Model();

	void train_batch(const std::vector<std::string>& filepath);

private:
	Conv2DLayer layer1_;
	Conv2DLayer layer2_;
	Conv2DLayer layer3_;
};



#endif //IMAGE_CLASSIFIER_MODEL_H
