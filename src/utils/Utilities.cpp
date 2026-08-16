//
// Created by fabian on 08/08/2026.
//

#include "../../include/utils/Utilities.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <zip.h>
#include <cstring>

namespace fs = std::filesystem;

bool Utilities::extract_zip(const std::string &archive_path, const std::string &extract_dir) {
	int err = 0;
	// Open the zip archive
	zip_t* archive = zip_open(archive_path.c_str(), 0, &err);
	if (!archive) {
		std::cerr << "Failed to open zip archive. Error code: " << err << "\n";
		return false;
	}

	// Ensure the target extraction directory exists
	fs::create_directories(extract_dir);

	// Get the number of files/directories inside the zip
	zip_int64_t num_entries = zip_get_num_entries(archive, 0);

	for (zip_int64_t i = 0; i < num_entries; ++i) {
		struct zip_stat st{};
		zip_stat_init(&st);
		zip_stat_index(archive, i, 0, &st);

		// Construct the output path
		fs::path target_path = fs::path(extract_dir) / st.name;

		// Check if the current entry is a directory (ends with '/')
		if (st.name[strlen(st.name) - 1] == '/') {
			fs::create_directories(target_path);
			continue;
		}

		// Ensure the parent directory for the current file exists
		fs::create_directories(target_path.parent_path());

		// Open the file inside the zip for reading
		zip_file_t* zf = zip_fopen_index(archive, i, 0);
		if (!zf) {
			std::cerr << "Failed to open file in zip: " << st.name << "\n";
			continue;
		}

		// Create the output file on disk
		std::ofstream out_file(target_path, std::ios::binary);
		if (!out_file.is_open()) {
			std::cerr << "Failed to create output file: " << target_path << "\n";
			zip_fclose(zf);
			continue;
		}

		// Read the file in chunks and write to disk
		char buffer[8192];
		zip_int64_t bytes_read;
		while ((bytes_read = zip_fread(zf, buffer, sizeof(buffer))) > 0) {
			out_file.write(buffer, bytes_read);
		}

		// Clean up current file
		out_file.close();
		zip_fclose(zf);
	}

	// Close the archive
	zip_close(archive);
	return true;
}
