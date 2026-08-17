//
// Created by fabian on 17/08/2026.
//

#include "../../include/model/Model.h"
#include "../../include/utils/Matrix3D.h"
#include "../../include/utils/DataLoader.h"
#include "../../include/layers/Conv2DLayer.h"

#include <filesystem>
#include <iostream>
#include <string>

struct Context {
	Matrix3D input;
	Matrix3D output1;
	Matrix3D output2;
	Matrix3D output3;
	Matrix3D activated;
};

Model::Model() : layer1_(16, 3, 3), layer2_(32, 3, 16), layer3_(64, 3, 32) {
}

void Model::train_batch(const std::vector<std::string>& file_paths) {
	std::vector<Context> batch_contexts(file_paths.size());

	try {
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < file_paths.size(); ++i) {
			try {
				constexpr int target_w = 128;
				constexpr int target_h = 128;
				Matrix3D tensor = DataLoader::load_image(file_paths[i], target_w, target_h);

				Matrix3D output_map1 = layer1_.forward_pass(tensor);
				Matrix3D output_map2 = layer2_.forward_pass(output_map1);
				Matrix3D output_map3 = layer3_.forward_pass(output_map2);
				Matrix3D activated_map = output_map3.apply_relu();

				batch_contexts[i] = Context{tensor, output_map1, output_map2, output_map3, activated_map};
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
