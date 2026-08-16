//
// Created by fabian on 08/08/2026.
//

#ifndef IMAGE_CLASSIFIER_UTILITIES_H
#define IMAGE_CLASSIFIER_UTILITIES_H
#include <string>


class Utilities {
public:
	static bool extract_zip(const std::string& archive_path, const std::string& extract_dir);
};



#endif //IMAGE_CLASSIFIER_UTILITIES_H
