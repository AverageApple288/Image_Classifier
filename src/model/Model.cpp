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

Model::Model() = default;

void Model::add_image(const std::string& path) {
	try {
		std::cout << "Processing image: " << path << "\n";

		constexpr int target_w = 128;
		constexpr int target_h = 128;
		const Matrix3D tensor = DataLoader::load_image(path, target_w, target_h);

		auto layer1 = Conv2DLayer(16, 3, 3);
		auto layer2 = Conv2DLayer(32, 3, 16);
		auto layer3 = Conv2DLayer(64, 3, 32);

		const Matrix3D output_map1 = layer1.forward_pass(tensor);
		const Matrix3D output_map2 = layer2.forward_pass(output_map1);
		const Matrix3D output_map3 = layer3.forward_pass(output_map2);

		this->output_feature_maps.push_back(output_map3);
	} catch (const std::exception& e) {
		std::cerr << "DataLoader test failed: " << e.what() << "\n";
	}
}
