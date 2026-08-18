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
	Matrix3D activated1;
	PoolResult pooling1;
	Matrix3D output2;
	Matrix3D activated2;
	PoolResult pooling2;
	Matrix3D output3;
	Matrix3D activated3;
	PoolResult pooling3;
};

Model::Model() : layer1_(16, 3, 3), layer2_(32, 3, 16), layer3_(64, 3, 32), pooling_layer1_(2, 2, 16), pooling_layer2_(2, 2, 32), pooling_layer3_(2, 2, 64) {
}

void Model::train_batch(const std::vector<std::string>& file_paths) const {
	std::vector<Context> batch_contexts(file_paths.size());

	try {
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < file_paths.size(); ++i) {
			try {
				constexpr int target_w = 128;
				constexpr int target_h = 128;
				Matrix3D tensor = DataLoader::load_image(file_paths[i], target_w, target_h);

				if (tensor.rows() < 3 || tensor.cols() < 3 || tensor.depth() == 0) {
					continue;
				}

				auto t0 = std::chrono::high_resolution_clock::now();

				// Block 1
				auto t1 = std::chrono::high_resolution_clock::now();
				Matrix3D output_map1 = layer1_.forward_pass(tensor);
				auto t2 = std::chrono::high_resolution_clock::now();
				Matrix3D activated_map1 = output_map1.apply_relu();
				auto t3 = std::chrono::high_resolution_clock::now();
				auto [output, mask] = pooling_layer1_.forward_pass(activated_map1);
				Matrix3D pooled_map1 = output;
				Matrix3D argmax_map1 = mask;
				auto t4 = std::chrono::high_resolution_clock::now();

				// Block 2
				Matrix3D output_map2 = layer2_.forward_pass(pooled_map1);
				auto t5 = std::chrono::high_resolution_clock::now();
				Matrix3D activated_map2 = output_map2.apply_relu();
				auto t6 = std::chrono::high_resolution_clock::now();
				auto [output2, mask2] = pooling_layer2_.forward_pass(activated_map2);
				Matrix3D pooled_map2 = output2;
				Matrix3D argmax_map2 = mask2;
				auto t7 = std::chrono::high_resolution_clock::now();

				// Block 3
				Matrix3D output_map3 = layer3_.forward_pass(pooled_map2);
				auto t8 = std::chrono::high_resolution_clock::now();
				Matrix3D activated_map3 = output_map3.apply_relu();
				auto t9 = std::chrono::high_resolution_clock::now();
				auto [output3, mask3] = pooling_layer3_.forward_pass(activated_map3);
				Matrix3D pooled_map3 = output3;
				Matrix3D argmax_map3 = mask3;
				auto t10 = std::chrono::high_resolution_clock::now();

				if (i == 0) {
					auto ms = [](auto start, auto end) {
						return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					};

					std::cout << "\n=== Profile Sample 0 ===" << std::endl;
					std::cout << "Image Loading:  " << ms(t0, t1) << " ms" << std::endl;
					std::cout << "Block 1 Conv:   " << ms(t1, t2) << " ms | ReLU: " << ms(t2, t3) << " ms | Pool: " << ms(t3, t4) << " ms" << std::endl;
					std::cout << "Block 2 Conv:   " << ms(t4, t5) << " ms | ReLU: " << ms(t5, t6) << " ms | Pool: " << ms(t6, t7) << " ms" << std::endl;
					std::cout << "Block 3 Conv:   " << ms(t7, t8) << " ms | ReLU: " << ms(t8, t9) << " ms | Pool: " << ms(t9, t10) << " ms" << std::endl;
					std::cout << "========================\n" << std::endl;
				}

				batch_contexts[i] = Context{tensor, output_map1, activated_map1, pooled_map1, argmax_map1, output_map2, activated_map2, pooled_map2, argmax_map2, output_map3, activated_map3, pooled_map3, argmax_map3};
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
