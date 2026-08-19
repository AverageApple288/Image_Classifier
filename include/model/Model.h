//
// Created by fabian on 17/08/2026.
//

#ifndef IMAGE_CLASSIFIER_MODEL_H
#define IMAGE_CLASSIFIER_MODEL_H
#include <string>
#include <vector>

#include "../layers/Conv2DLayer.h"
#include "../layers/PoolingLayer.h"
#include "../layers/DenseLayer.h"

struct LabeledSample {
	std::string path;
	size_t label;
};

struct Context {
	Matrix3D input;
	// Layer 1
	Matrix3D output1, activated1, pooled1, argmax1;
	// Layer 2
	Matrix3D output2, activated2, pooled2, argmax2;
	// Layer 3
	Matrix3D output3, activated3, pooled3, argmax3;
	// Dense / Classification
	std::vector<float> dense_input;
	std::vector<float> logits;
	std::vector<float> probabilities;
	size_t label;
	float loss;
};

class Model {
public:
	Model();

	void train_batch(const std::vector<LabeledSample>& batch) const;

private:
	Conv2DLayer layer1_;
	Conv2DLayer layer2_;
	Conv2DLayer layer3_;
	PoolingLayer pooling_layer1_;
	PoolingLayer pooling_layer2_;
	PoolingLayer pooling_layer3_;
	DenseLayer dense_layer_;
};



#endif //IMAGE_CLASSIFIER_MODEL_H
