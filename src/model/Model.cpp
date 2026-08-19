//
// Created by fabian on 17/08/2026.
//

#include "../../include/model/Model.h"

#include <cmath>

#include "../../include/utils/Matrix3D.h"
#include "../../include/utils/DataLoader.h"
#include "../../include/layers/Conv2DLayer.h"
#include "../../include/model/LossFunction.h"

#include <filesystem>
#include <iostream>

Model::Model() : layer1_(16, 3, 3), layer2_(32, 3, 16), layer3_(64, 3, 32), pooling_layer1_(2, 2, 16), pooling_layer2_(2, 2, 32), pooling_layer3_(2, 2, 64), dense_layer_(12544, 2) {
}

void Model::train_batch(const std::vector<LabeledSample>& batch) {
	std::vector<Context> batch_contexts(batch.size());
	float batch_loss = 0.0f;

	try {
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < batch.size(); ++i) {
			try {
				const auto&[path, label] = batch[i];

				constexpr int target_w = 128;
				constexpr int target_h = 128;
				Matrix3D tensor = DataLoader::load_image(path, target_w, target_h);

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

				batch_contexts[i] = Context{
					tensor,
					output_map1, activated_map1, pooled_map1, argmax_map1,
					output_map2, activated_map2, pooled_map2, argmax_map2,
					output_map3, activated_map3, pooled_map3, argmax_map3,
					flat_features,
					logits,
					label
				};
			} catch (const std::exception& e) {
				#pragma omp critical
				{
					std::cerr << "Error processing image in batch: " << e.what() << "\n";
				}
			}
		}

		for (size_t i = 0; i < batch.size(); ++i) {
			Context& ctx = batch_contexts[i];

			std::vector<float> probabilities = LossFunction::softmax(ctx.logits);

			float loss = LossFunction::cross_entropy(probabilities[ctx.label]);
			batch_loss += loss;

			std::vector<float> d_logits = probabilities;
			d_logits[ctx.label] -= 1.0f;

			std::vector<float> d_dense_input = dense_layer_.backward_pass(d_logits, ctx.dense_input);

			const size_t d_depth = ctx.pooled3.depth();
			const size_t d_rows = ctx.pooled3.rows();
			const size_t d_cols = ctx.pooled3.cols();

			Matrix3D d_pool3_output(d_depth, d_rows, d_cols);

			size_t flat_index = 0;

			for (size_t d = 0; d < d_depth; ++d) {
				for (size_t r = 0; r < d_rows; ++r) {
					for (size_t c = 0; c < d_cols; ++c) {
						d_pool3_output(d, r, c) = d_dense_input[flat_index++];
					}
				}
			}

			Matrix3D d_conv3_output = pooling_layer3_.backward_pass(d_pool3_output, ctx.argmax3);
			Matrix3D d_conv3_input = layer3_.backward_pass(Matrix3D::apply_relu_derivative(d_conv3_output, ctx.output3), ctx.pooled2);

			Matrix3D d_conv2_output = pooling_layer2_.backward_pass(d_conv3_input, ctx.argmax2);
			Matrix3D d_conv2_input  = layer2_.backward_pass(Matrix3D::apply_relu_derivative(d_conv2_output, ctx.output2), ctx.pooled1);

			Matrix3D d_conv1_output = pooling_layer1_.backward_pass(d_conv2_input, ctx.argmax1);
			Matrix3D d_input_tensor = layer1_.backward_pass(Matrix3D::apply_relu_derivative(d_conv1_output, ctx.output1), ctx.input);
		}

		float avg_batch_loss = batch_loss / batch.size();
		std::cout << "Batch Loss: " << avg_batch_loss << std::endl;

		float learning_rate = 0.001f;
		dense_layer_.update_weights(learning_rate, batch.size());

		layer3_.update_weights(learning_rate, batch.size());

		layer2_.update_weights(learning_rate, batch.size());

		layer1_.update_weights(learning_rate, batch.size());
	} catch (const std::exception& e) {
		std::cerr << "DataLoader test failed: " << e.what() << "\n";
	}
}
