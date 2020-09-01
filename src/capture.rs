use opencv::{prelude::*, videoio};

type Result<T> = opencv::Result<T>;

pub(crate) struct Capture {
    capture: videoio::VideoCapture,
}

impl Capture {
    pub fn create_default(width: i32, height: i32) -> Result<Self> {
        Self::create(0, width, height)
    }

    pub fn create(index: i32, width: i32, height: i32) -> Result<Self> {
        let mut capture = videoio::VideoCapture::new(index, videoio::CAP_ANY)?;
        capture.set(videoio::CAP_PROP_FRAME_WIDTH, width as f64)?;
        capture.set(videoio::CAP_PROP_FRAME_HEIGHT, height as f64)?;
        Ok(Self { capture })
    }

    pub fn is_opened(&self) -> Result<bool> {
        videoio::VideoCapture::is_opened(&self.capture)
    }

    pub fn grab_frame(&mut self) -> Result<Option<Mat>> {
        if !self.capture.grab()? {
            return Ok(None);
        }

        let mut frame = Mat::default()?;
        self.capture.retrieve(&mut frame, 0)?;
        Ok(Some(frame))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.capture.release();
    }
}
