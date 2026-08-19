//
// Created by fabian on 16/08/2026.
//

#ifndef IMAGE_CLASSIFIER_DENSELAYER_H
#define IMAGE_CLASSIFIER_DENSELAYER_H
#include <vector>


class DenseLayer {
public:
	DenseLayer(size_t num_inputs, size_t num_outputs);

	[[nodiscard]] std::vector<float> forward_pass(const std::vector<float>& input) const;

private:
	size_t num_inputs_;
	size_t num_outputs_;

	std::vector<float> weights_;
	std::vector<float> biases_;
};



#endif //IMAGE_CLASSIFIER_DENSELAYER_H
