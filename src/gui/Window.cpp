#include "../../include/gui/Window.h"

#include <filesystem>
#include <iostream>
#include <thread>
#include <omp.h>

#include "../../include/model/Model.h"
#include "../../include/utils/Utilities.h"

Window::Window() {
	set_title("Image Classifier");
    set_default_size(800, 600); // Increased size to fit the new UI

    // MAIN CONTAINER
    main_vbox_.set_orientation(Gtk::Orientation::VERTICAL);
    main_vbox_.set_spacing(20);
    main_vbox_.set_margin(40);
    set_child(main_vbox_);

    // TITLE & DESCRIPTION
    title_label_.set_markup("<span size='30000' weight='bold'>Image Classifier</span>");
    main_vbox_.append(title_label_);

    desc_label_.set_text("This is a simple image classifier I made from scratch (without importing any artificial intelligence packages) as a small project. You can try it out by uploading some images below to train the model and then test it's classification skills.");
    desc_label_.set_wrap(true);
    desc_label_.set_justify(Gtk::Justification::CENTER);
    main_vbox_.append(desc_label_);

    // THE "CARD" CONTAINER (Holds the form)
    card_vbox_.set_orientation(Gtk::Orientation::VERTICAL);
    card_vbox_.set_spacing(15);
    card_vbox_.set_margin(20);
    // Note: To get the dark grey background for the card, you will need to apply a CSS class here.
    main_vbox_.append(card_vbox_);

    // THE TWO COLUMNS (Horizontal Box)
    cols_hbox_.set_orientation(Gtk::Orientation::HORIZONTAL);
    cols_hbox_.set_spacing(30);
    cols_hbox_.set_homogeneous(true); // Makes left and right columns equal width
    card_vbox_.append(cols_hbox_);

    // --- LEFT COLUMN (Dataset One) ---
    left_vbox_.set_orientation(Gtk::Orientation::VERTICAL);
    left_vbox_.set_spacing(10);

    left_label_.set_text("Upload a zip file of images for\ndataset one:");
    left_label_.set_halign(Gtk::Align::START); // Align text to the left
    left_vbox_.append(left_label_);

    // Browse row
    left_browse_hbox_.set_orientation(Gtk::Orientation::HORIZONTAL);
    left_browse_hbox_.set_spacing(10);
    left_browse_btn_.set_label("Browse...");
    left_file_status_.set_text("No file selected.");
    left_browse_hbox_.append(left_browse_btn_);
    left_browse_hbox_.append(left_file_status_);
    left_vbox_.append(left_browse_hbox_);

    left_entry_.set_placeholder_text("Dataset one name");
    left_vbox_.append(left_entry_);

    cols_hbox_.append(left_vbox_);

    // --- RIGHT COLUMN (Dataset Two) ---
    right_vbox_.set_orientation(Gtk::Orientation::VERTICAL);
    right_vbox_.set_spacing(10);

    right_label_.set_text("Upload a zip file of images for\ndataset two:");
    right_label_.set_halign(Gtk::Align::START);
    right_vbox_.append(right_label_);

    // Browse row
    right_browse_hbox_.set_orientation(Gtk::Orientation::HORIZONTAL);
    right_browse_hbox_.set_spacing(10);
    right_browse_btn_.set_label("Browse...");
    right_file_status_.set_text("No file selected.");
    right_browse_hbox_.append(right_browse_btn_);
    right_browse_hbox_.append(right_file_status_);
    right_vbox_.append(right_browse_hbox_);

    right_entry_.set_placeholder_text("Dataset two name");
    right_vbox_.append(right_entry_);

    cols_hbox_.append(right_vbox_);

    // UPLOAD BUTTON (Bottom of the card)
    upload_btn_.set_label("Upload");
    card_vbox_.append(upload_btn_);

	// Connect left browse button
	left_browse_btn_.signal_clicked().connect(
		sigc::mem_fun(*this, &Window::on_left_browse_clicked)
	);

	// Connect right browse button
	right_browse_btn_.signal_clicked().connect(
		sigc::mem_fun(*this, &Window::on_right_browse_clicked)
	);

	// Connect upload button
	upload_btn_.signal_clicked().connect(
		sigc::mem_fun(*this, &Window::on_upload_button_clicked)
	);

	progress_bar_.set_show_text(true);
	progress_bar_.set_fraction(0.0);
	progress_bar_.set_visible(false);
	card_vbox_.append(progress_bar_);

	dispatcher_.connect(sigc::mem_fun(*this, &Window::update_progress_ui));
}

void Window::on_left_browse_clicked() {
	const auto dialog = Gtk::FileDialog::create();
	dialog->set_title("Select Dataset One (ZIP)");

	// Create a list store to hold your filters
	const auto filters = Gio::ListStore<Gtk::FileFilter>::create();

	// Create the specific ZIP filter
	const auto filter_zip = Gtk::FileFilter::create();
	filter_zip->set_name("ZIP Archives");
	filter_zip->add_pattern("*.zip");

	// Add the filter to the list and apply it to the dialog
	filters->append(filter_zip);
	dialog->set_filters(filters);
	dialog->set_default_filter(filter_zip);

	dialog->open(*this, sigc::bind(sigc::mem_fun(*this, &Window::on_left_file_dialog_finish), dialog));
}

void Window::on_left_file_dialog_finish(const Glib::RefPtr<Gio::AsyncResult>& result, const Glib::RefPtr<Gtk::FileDialog>& dialog) {
	try {
		if (const auto file = dialog->open_finish(result)) {
			const std::string filename = file->get_basename();
			const std::string full_path = file->get_path(); // We need the full path to extract

			// Update the UI
			left_file_status_.set_text(filename);

			// Define your relative path (e.g., a 'datasets' folder in your build directory)
			const std::string relative_extract_path = "./datasets/dataset_one";

			std::cout << "Extracting " << filename << " to " << relative_extract_path << "...\n";

			// Run the extraction
			if (Utilities::extract_zip(full_path, relative_extract_path)) {
				std::cout << "Dataset one extracted successfully!\n";
			} else {
				std::cerr << "Failed to extract dataset one.\n";
			}
		}
	} catch (const Glib::Error& err) {
		// This catches cases where the user clicks "Cancel" or closes the dialog
		std::cout << "File selection cancelled or failed: " << err.what() << std::endl;
	}
}

void Window::on_right_browse_clicked() {
	const auto dialog = Gtk::FileDialog::create();
	dialog->set_title("Select Dataset Two (ZIP)");

	// Apply the same filter logic for the right button
	const auto filters = Gio::ListStore<Gtk::FileFilter>::create();
	const auto filter_zip = Gtk::FileFilter::create();
	filter_zip->set_name("ZIP Archives");
	filter_zip->add_pattern("*.zip");

	filters->append(filter_zip);
	dialog->set_filters(filters);
	dialog->set_default_filter(filter_zip);

	dialog->open(*this, sigc::bind(sigc::mem_fun(*this, &Window::on_right_file_dialog_finish), dialog));
}

void Window::on_right_file_dialog_finish(const Glib::RefPtr<Gio::AsyncResult>& result, const Glib::RefPtr<Gtk::FileDialog>& dialog) {
	try {
		if (const auto file = dialog->open_finish(result)) {
			const std::string filename = file->get_basename();
			const std::string full_path = file->get_path(); // We need the full path to extract

			// Update the UI
			right_file_status_.set_text(filename);

			// Define your relative path (e.g., a 'datasets' folder in your build directory)
			const std::string relative_extract_path = "./datasets/dataset_two";

			std::cout << "Extracting " << filename << " to " << relative_extract_path << "...\n";

			// Run the extraction
			if (Utilities::extract_zip(full_path, relative_extract_path)) {
				std::cout << "Dataset two extracted successfully!\n";
			} else {
				std::cerr << "Failed to extract dataset one.\n";
			}
		}
	} catch (const Glib::Error& err) {
		std::cout << "File selection cancelled or failed: " << err.what() << std::endl;
	}
}

void Window::on_upload_button_clicked() {
	upload_btn_.set_sensitive(false);
	left_browse_btn_.set_sensitive(false);
	right_browse_btn_.set_sensitive(false);

	progress_bar_.set_fraction(0.0);
	progress_bar_.set_visible(true);

	std::thread worker_thread([this] {
		int total_cores = std::thread::hardware_concurrency();
		int threads_to_use = std::max(1, total_cores - 2);
		omp_set_num_threads(threads_to_use);

		try {
			auto model = Model();

			const std::string dataset1_dir = "./datasets/dataset_one";
			const std::string dataset2_dir = "./datasets/dataset_two";
			std::vector<std::string> image_paths;

			// Collect all valid file paths sequentially
			if (std::filesystem::exists(dataset1_dir)) {
				for (const auto& entry : std::filesystem::recursive_directory_iterator(dataset1_dir)) {
					if (entry.is_regular_file()) {
						if (std::string ext = entry.path().extension().string(); ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
							image_paths.push_back(entry.path().string());
						}
					}
				}
			}

			size_t total_images = image_paths.size();
			size_t processed_images = 0;

			constexpr size_t batch_size = 32;

			for (size_t i = 0; i < image_paths.size(); i += batch_size) {
				const size_t current_batch_size = std::min(batch_size, image_paths.size() - i);

				std::vector batch_paths(
					image_paths.begin() + i,
					image_paths.begin() + i + current_batch_size
				);

				model.train_batch(batch_paths);

				processed_images += current_batch_size;
				progress_fraction_ = static_cast<double>(processed_images) / total_images;

				dispatcher_.emit();
			}

			progress_bar_.set_visible(false);

			std::cout << "Finished first convolution step" << std::endl;
		} catch (const std::exception& e) {
			std::cerr << "Error in background thread: " << e.what() << std::endl;
		}
	});

	worker_thread.detach();
}

void Window::update_progress_ui() {
	progress_bar_.set_fraction(progress_fraction_.load());
	progress_bar_.set_text(std::to_string(static_cast<int>(progress_fraction_.load() * 100.0)) + "%");
}

Window::~Window() {
	try {
		// Check if the directory exists to avoid unnecessary errors
		if (const std::filesystem::path dataset_dir = "./datasets"; std::filesystem::exists(dataset_dir)) {
			// remove_all recursively deletes the folder and all its contents
			std::filesystem::remove_all(dataset_dir);
			std::cout << "Cleaned up temporary dataset files successfully.\n";
		}
	} catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "Error cleaning up datasets on close: " << e.what() << "\n";
	}
}