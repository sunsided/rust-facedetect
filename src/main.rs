extern crate opencv;
use opencv::core::{Rect, Scalar, Size};
use opencv::{highgui, imgproc, objdetect, prelude::*, types, videoio};

type Result<T> = opencv::Result<T>;

const WINDOW_NAME: &str = "OpenCV Face Detection in Rust";
const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const SCALE_FACTOR: f64 = 0.25f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

struct Window<'a> {
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

fn run() -> Result<()> {
    let mut classifier = objdetect::CascadeClassifier::new(CASCADE_XML_FILE)?;

    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    capture.set(videoio::CAP_PROP_FRAME_WIDTH, 800f64)?;
    capture.set(videoio::CAP_PROP_FRAME_HEIGHT, 600f64)?;

    let opened = videoio::VideoCapture::is_opened(&capture)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let window = Window::create(WINDOW_NAME, 800, 600)?;

    run_main_loop(&mut capture, &mut classifier, &window)?;

    capture.release()?;
    Ok(())
}

fn run_main_loop(
    mut capture: &mut videoio::VideoCapture,
    mut classifier: &mut objdetect::CascadeClassifier,
    window: &Window,
) -> Result<()> {
    loop {
        const KEY_CODE_ESCAPE: i32 = 27;
        if let Ok(KEY_CODE_ESCAPE) = highgui::wait_key(10) {
            return Ok(());
        }

        let mut frame = match grab_frame(&mut capture)? {
            Some(frame) => frame,
            None => continue,
        };

        let mut preprocessed = preprocess_image(&mut frame)?;
        let faces = detect_faces(&mut classifier, &mut preprocessed)?;
        for face in faces {
            draw_box_around_face(&mut frame, face)?;
        }

        window.show_image(&frame)?;
    }
}

fn grab_frame(capture: &mut videoio::VideoCapture) -> Result<Option<Mat>> {
    if !capture.grab()? {
        return Ok(None);
    }

    let mut frame = Mat::default()?;
    capture.retrieve(&mut frame, 0)?;
    Ok(Some(frame))
}

fn preprocess_image(frame: &mut Mat) -> Result<Mat> {
    let mut gray = Mat::default()?;
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut reduced = Mat::default()?;
    imgproc::resize(
        &gray,
        &mut reduced,
        Size {
            width: 0,
            height: 0,
        },
        SCALE_FACTOR,
        SCALE_FACTOR,
        imgproc::INTER_LINEAR,
    )?;

    let mut equalized = Mat::default()?;
    imgproc::equalize_hist(&reduced, &mut equalized)?;

    Ok(equalized)
}

fn detect_faces(
    classifier: &mut objdetect::CascadeClassifier,
    image: &mut Mat,
) -> Result<types::VectorOfRect> {
    const SCALE_FACTOR: f64 = 1.1;
    const MIN_NEIGHBORS: i32 = 2;
    const FLAGS: i32 = 0;
    const MIN_FACE_SIZE: Size = Size {
        width: 30,
        height: 30,
    };
    const MAX_FACE_SIZE: Size = Size {
        width: 0,
        height: 0,
    };

    let mut faces = types::VectorOfRect::new();
    classifier.detect_multi_scale(
        image,
        &mut faces,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        FLAGS,
        MIN_FACE_SIZE,
        MAX_FACE_SIZE,
    )?;
    Ok(faces)
}

fn draw_box_around_face(frame: &mut Mat, face: Rect) -> Result<()> {
    println!("found face {:?}", face);
    let scaled_face = Rect {
        x: face.x * SCALE_FACTOR_INV,
        y: face.y * SCALE_FACTOR_INV,
        width: face.width * SCALE_FACTOR_INV,
        height: face.height * SCALE_FACTOR_INV,
    };

    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;
    let color_red = Scalar::new(0f64, 0f64, 255f64, -1f64);

    imgproc::rectangle(frame, scaled_face, color_red, THICKNESS, LINE_TYPE, SHIFT)?;
    Ok(())
}

fn main() {
    run().unwrap()
}
