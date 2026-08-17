//
// Created by fabian on 15/01/2026.
//

#ifndef IMAGE_CLASSIFIER_WINDOW_H
#define IMAGE_CLASSIFIER_WINDOW_H
#include <gtkmm.h>


class Window : public Gtk::Window {
public:
	Window();
	~Window() override;

protected:
	// --- Containers ---
	Gtk::Box main_vbox_;
	Gtk::Box card_vbox_;
	Gtk::Box cols_hbox_;

	Gtk::Box left_vbox_;
	Gtk::Box left_browse_hbox_;

	Gtk::Box right_vbox_;
	Gtk::Box right_browse_hbox_;

	// --- Text/Labels ---
	Gtk::Label title_label_;
	Gtk::Label desc_label_;

	Gtk::Label left_label_;
	Gtk::Label left_file_status_;

	Gtk::Label right_label_;
	Gtk::Label right_file_status_;

	// --- Inputs & Buttons ---
	Gtk::Button left_browse_btn_;
	Gtk::Entry left_entry_;

	Gtk::Button right_browse_btn_;
	Gtk::Entry right_entry_;

	Gtk::Button upload_btn_;

	void on_left_browse_clicked();
	void on_right_browse_clicked();

	// --- Async Callbacks for FileDialog ---
	void on_left_file_dialog_finish(const Glib::RefPtr<Gio::AsyncResult>& result, const Glib::RefPtr<Gtk::FileDialog>& dialog);
	void on_right_file_dialog_finish(const Glib::RefPtr<Gio::AsyncResult>& result, const Glib::RefPtr<Gtk::FileDialog>& dialog);

	void on_upload_button_clicked();
};


#endif //IMAGE_CLASSIFIER_WINDOW_H