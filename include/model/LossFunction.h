//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_LOSSFUNCTION_H
#define IMAGE_CLASSIFIER_LOSSFUNCTION_H
#include <vector>


class LossFunction {
public:
	static std::vector<float> softmax(const std::vector<float>& input);

	static float cross_entropy(float predicted_probability);
};



#endif //IMAGE_CLASSIFIER_LOSSFUNCTION_H
