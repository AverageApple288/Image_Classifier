//
// Created by fabian on 17/08/2026.
//

#include "../../include/model/Model.h"
#include "../../include/utils/Matrix3D.h"
#include "../../include/utils/DataLoader.h"
#include "../../include/layers/Conv2DLayer.h"
#include "../../include/model/LossFunction.h"

#include <filesystem>
#include <iostream>

Model::Model() : layer1_(16, 3, 3), layer2_(32, 3, 16), layer3_(64, 3, 32), pooling_layer1_(2, 2, 16), pooling_layer2_(2, 2, 32), pooling_layer3_(2, 2, 64), dense_layer_(16348, 2) {
}

void Model::train_batch(const std::vector<LabeledSample>& batch) const {
	std::vector<Context> batch_contexts(batch.size());

	try {
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < batch.size(); ++i) {
			try {
				const auto&[path, label] = batch[i];

				constexpr int target_w = 128;
				constexpr int target_h = 128;
				Matrix3D tensor = DataLoader::load_image(path, target_w, target_h);

				if (tensor.rows() < 3 || tensor.cols() < 3 || tensor.depth() == 0) {
					continue;
				}

				// Block 1
				Matrix3D output_map1 = layer1_.forward_pass(tensor);
				Matrix3D activated_map1 = output_map1.apply_relu();
				auto [output, mask] = pooling_layer1_.forward_pass(activated_map1);
				Matrix3D pooled_map1 = output;
				Matrix3D argmax_map1 = mask;

				// Block 2
				Matrix3D output_map2 = layer2_.forward_pass(pooled_map1);
				Matrix3D activated_map2 = output_map2.apply_relu();
				auto [output2, mask2] = pooling_layer2_.forward_pass(activated_map2);
				Matrix3D pooled_map2 = output2;
				Matrix3D argmax_map2 = mask2;

				// Block 3
				Matrix3D output_map3 = layer3_.forward_pass(pooled_map2);
				Matrix3D activated_map3 = output_map3.apply_relu();
				auto [output3, mask3] = pooling_layer3_.forward_pass(activated_map3);
				Matrix3D pooled_map3 = output3;
				Matrix3D argmax_map3 = mask3;

				const std::vector<float>& flat_features = pooled_map3.get_flat_data();

				const std::vector<float>& logits = dense_layer_.forward_pass(flat_features);

				const std::vector<float> probabilities = LossFunction::softmax(logits);

				float correct_class_probability = probabilities[label];

				float loss = LossFunction::cross_entropy(correct_class_probability);

				batch_contexts[i] = Context{
					tensor,
					output_map1, activated_map1, pooled_map1, argmax_map1,
					output_map2, activated_map2, pooled_map2, argmax_map2,
					output_map3, activated_map3, pooled_map3, argmax_map3,
					flat_features,
					logits,
					probabilities,
					label,
					loss
				};
			} catch (const std::exception& e) {
				#pragma omp critical
				{
					std::cerr << "Error processing image in batch: " << e.what() << "\n";
				}
			}
		}
	} catch (const std::exception& e) {
		std::cerr << "DataLoader test failed: " << e.what() << "\n";
	}
}
