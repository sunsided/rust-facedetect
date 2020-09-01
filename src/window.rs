use opencv::{highgui, prelude::*};

type Result<T> = opencv::Result<T>;

pub(crate) struct Window<'a> {
    name: &'a str,
}

impl<'a> Window<'a> {
    pub fn create(name: &'a str, width: i32, height: i32) -> Result<Self> {
        highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO)?;
        highgui::resize_window(name, width, height)?;
        Ok(Self { name })
    }

    pub fn show_image(&self, frame: &Mat) -> Result<()> {
        highgui::imshow(&self.name, &frame)?;
        Ok(())
    }
}

impl<'a> Drop for Window<'a> {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(self.name);
    }
}
