//
// Created by fabian on 16/08/2026.
//

#include "../../include/layers/DenseLayer.h"

#include <iostream>
#include <ostream>
#include <random>

DenseLayer::DenseLayer(const size_t num_inputs, const size_t num_outputs) : num_inputs_(num_inputs), num_outputs_(num_outputs) {
	std::random_device rd;
	std::mt19937 gen(rd());
	const float limit = std::sqrt(2.0f / static_cast<float>(num_inputs_));
	std::uniform_real_distribution dis(-limit, limit);

	weights_.resize(num_inputs_ * num_outputs_);
	biases_.assign(num_outputs_, 0.0f);

	for (size_t i = 0; i < num_inputs_ * num_outputs_; ++i) {
		weights_[i] = dis(gen);
	}

	d_weights_.assign(num_inputs_ * num_outputs_, 0.0f);
	d_biases_.assign(num_outputs_, 0.0f);
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

std::vector<float> DenseLayer::backward_pass(const std::vector<float> &d_logits, const std::vector<float> &cached_input) {
	std::vector d_input(num_inputs_, 0.0f);

	for (size_t i = 0; i < num_outputs_; ++i) {
		d_biases_[i] += d_logits[i];
		for (size_t j = 0; j < num_inputs_; ++j) {
			d_weights_[i * num_inputs_ + j] += d_logits[i] * cached_input[j];
			d_input[j] += d_logits[i] * weights_[i * num_inputs_ + j];
		}
	}

	return d_input;
}

void DenseLayer::update_weights(const float learning_rate, const size_t batch_size) {
	const float scale = learning_rate / static_cast<float>(batch_size);

	for (size_t i = 0; i < weights_.size(); ++i) {
		weights_[i] -= scale * d_weights_[i];
		d_weights_[i] = 0.0f;
	}

	for (size_t i = 0; i < biases_.size(); ++i) {
		biases_[i] -= scale * d_biases_[i];
		d_biases_[i] = 0.0f;
	}
}