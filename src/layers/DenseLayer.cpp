//
// Created by fabian on 16/08/2026.
//

#include "../../include/layers/DenseLayer.h"

#include <random>

DenseLayer::DenseLayer(const size_t num_inputs, const size_t num_outputs) : num_inputs_(num_inputs), num_outputs_(num_outputs) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.1, 0.1);

	weights_.resize(num_inputs_ * num_outputs_);
	biases_.resize(num_outputs_);

	for (size_t i = 0; i < num_inputs_ * num_outputs_; ++i) {
		weights_[i] = dis(gen);
	}
}

std::vector<float> DenseLayer::forward_pass(const std::vector<float> &input) const {
	std::vector<float> output;
	output.resize(num_outputs_);

	for (size_t i = 0; i < num_outputs_; ++i) {
		for (size_t j = 0; j < num_inputs_; ++j) {
			output[i] += weights_[i * num_inputs_ + j] * input[j];
		}

		output[i] += biases_[i];
	}

	return output;
}