#include <iostream>

#include "include/gui/Window.h"

int main(int argc, char* argv[])
{
	// 8. SETUP: Create the application object
	// "org.gtkmm.example" must be a unique ID for your app
	auto app = Gtk::Application::create("org.gtkmm.example");

	// Run the application, showing your custom window
	return app->make_window_and_run<Window>(argc, argv);
}