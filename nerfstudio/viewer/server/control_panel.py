class ControlPanel:
    def __init__(self):
        pas
    def add_element(self, e):
        if e.name not in self.control_elements:
            self.control_elements[e.name] = e
        e.install(self.viser_server)

    def setup_default_control(self):
        # train speed
        self.add_element(ViewerDropdown("Train Speed", "Balanced", ["Fast", "Balanced", "Slow"]))
        # output render
        self.add_element(ViewerDropdown("Output Render", "rgb", ["rgb"]))
        # colormap
        self.add_element(ViewerDropdown("Colormap", "default", ["default"], cb_hook=self.update_control_panel))
        if self.output_options.value != "rgb":
            with self.viser_server.gui_folder("Colormap Options"):
                self.add_element(ViewerCheckbox("Invert", False))
                self.add_element(ViewerCheckbox("Normalize", False))
                self.add_element(ViewerNumber("Min", 0.0))
                self.add_element(ViewerNumber("Max", 1.0))
        # train util
        self.add_element(ViewerSlider("Train Util", 0.9, 0, 1, 0.05))
        # max res
        self.add_element(ViewerSlider("Max Res.", 500, 100, 2000, 100))
        # crop viewport
        self.add_element(ViewerCheckbox("Crop Viewport", False, cb_hook=self.update_control_panel))
        if self.crop_viewport.value:
            with self.viser_server.gui_folder("Crop Options"):
                pass

    def teardown_control_panel(self):
        for _, e in self.control_elements:
            e.remove()

    def update_control_panel(self):
        self.teardown_control_panel()
        self.setup_default_control()