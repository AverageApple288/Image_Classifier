#include <iostream>
#include <fstream>
#include <filesystem>

#include "include/crow_all.h"
#include "include/ImageProcessor.h"

namespace fs = std::filesystem;

int main() {
    crow::SimpleApp app;

	if (std::filesystem::exists("uploads")) {
		std::filesystem::remove_all("uploads"); // Deletes folder + all contents
		std::cout << "Cleaned up old uploads." << std::endl;
	}

    // Ensure "uploads" exists
    fs::create_directories("uploads");

	ImageProcessor processor(32, 32);

    CROW_ROUTE(app, "/")([](){
        auto page = crow::mustache::load("index.html");
        return page.render();
    });

    CROW_ROUTE(app, "/upload").methods(crow::HTTPMethod::POST)([](const crow::request& req){
        crow::multipart::message file_message(req);

        std::string dataset1_name, dataset2_name;

        for (const auto& part : file_message.parts) {
            auto content_disposition = part.get_header_object("Content-Disposition");
            std::string input_name = content_disposition.params["name"];

            // 1. Handle Text Inputs
            if (input_name == "dataset1_name") dataset1_name = part.body;
            else if (input_name == "dataset2_name") dataset2_name = part.body;

            // 2. Handle File Uploads
            else if (input_name == "zipfile" || input_name == "zipfile2") {
                std::string filename = content_disposition.params["filename"];
                if (filename.empty()) continue;

                // A. Setup Paths
                // We create a specific folder for this dataset (e.g., uploads/dataset1_name/)
                // If the user didn't name the dataset yet, we fallback to the input name
                std::string target_folder = "uploads/" + input_name;
                fs::create_directories(target_folder);

                std::string filepath = target_folder + "/" + filename;

                // B. Save the Archive File
                std::ofstream out_file(filepath, std::ios::binary);
                out_file << part.body;
                out_file.close();

                std::cout << "Saved archive: " << filepath << std::endl;

                // C. Detect Format and Unzip
                std::string command;

                // Check extensions
                if (filename.find(".zip") != std::string::npos) {
                    // UNZIP: -o (overwrite), -d (destination)
                    command = "unzip -o \"" + filepath + "\" -d \"" + target_folder + "\"";
                }
                else if (filename.find(".tar.gz") != std::string::npos) {
                    // TAR GZ: -x (extract), -z (gzip), -f (file), -C (destination)
                    command = "tar -xzf \"" + filepath + "\" -C \"" + target_folder + "\"";
                }
                else if (filename.find(".tar.xz") != std::string::npos) {
                    // TAR XZ: -x (extract), -J (xz), -f (file), -C (destination)
                    command = "tar -xJf \"" + filepath + "\" -C \"" + target_folder + "\"";
                }

                // D. Execute Command
                if (!command.empty()) {
                    int result = std::system(command.c_str());
                    if (result == 0) std::cout << "Extracted successfully!\n";
                    else std::cerr << "Extraction failed for " << filename << "\n";
                }
            }
        }

    	std::string trainbutton = "<a href='/train' id='trainButton' style='width:75%'><button>Train Model</button></a>";

    	crow::mustache::context ctx;
    	ctx["trainbutton"] = trainbutton;

    	auto page = crow::mustache::load("index.html");

        return crow::response(page.render(ctx));
    });

	CROW_ROUTE(app, "/train").methods(crow::HTTPMethod::POST)([&processor](){

		std::vector<Matrix> dataset_1_imgs;
		std::string path = "uploads/zipfile";

		for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
			if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {

				// CLEAN USE: Just ask the object to do the work
				Matrix m = processor.load_and_process(entry.path().string());

				dataset_1_imgs.push_back(m);
			}
		}

		return crow::response("Processed " + std::to_string(dataset_1_imgs.size()) + " images.");
	});

    app.port(8080).multithreaded().run();
}