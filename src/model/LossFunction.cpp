//
// Created by fabian on 16/08/2026.
//

#include "../../include/model/LossFunction.h"

#include <algorithm>
#include <cmath>

std::vector<float> LossFunction::softmax(const std::vector<float> &input) {
	std::vector<float> output(input.size());
	const float max = *std::ranges::max_element(input);

	float sum = 0.0f;
	for (size_t i = 0; i < input.size(); ++i) {
		output[i] = std::exp(input[i] - max);
		sum += output[i];
	}

	for (size_t i = 0; i < input.size(); ++i) {
		output[i] /= sum;
	}

	return output;
}

float LossFunction::cross_entropy(const float predicted_probability) {
	constexpr float epsilon = 1e-7f;

	return -std::log(std::clamp(predicted_probability, epsilon, 1.0f));
}
