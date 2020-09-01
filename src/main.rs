mod capture;
mod window;

extern crate opencv;
use crate::capture::Capture;
use crate::window::Window;
use opencv::core::{Rect, Scalar, Size};
use opencv::{highgui, imgproc, objdetect, prelude::*, types};

type Result<T> = opencv::Result<T>;

const WINDOW_NAME: &str = "OpenCV Face Detection in Rust";
const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const CAPTURE_WIDTH: i32 = 800;
const CAPTURE_HEIGHT: i32 = 600;

const SCALE_FACTOR: f64 = 0.25f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

fn run() -> Result<()> {
    let mut classifier = objdetect::CascadeClassifier::new(CASCADE_XML_FILE)?;

    let mut capture = Capture::create_default(CAPTURE_WIDTH, CAPTURE_HEIGHT)?;
    let opened = capture.is_opened()?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let window = Window::create(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)?;

    run_main_loop(&mut capture, &mut classifier, &window)?;

    Ok(())
}

fn run_main_loop(
    capture: &mut Capture,
    classifier: &mut objdetect::CascadeClassifier,
    window: &Window,
) -> Result<()> {
    loop {
        const KEY_CODE_ESCAPE: i32 = 27;
        if let Ok(KEY_CODE_ESCAPE) = highgui::wait_key(10) {
            return Ok(());
        }

        let mut frame = match capture.grab_frame()? {
            Some(frame) => frame,
            None => continue,
        };

        let preprocessed = preprocess_image(&frame)?;
        let faces = detect_faces(classifier, preprocessed)?;
        for face in faces {
            draw_box_around_face(&mut frame, face)?;
        }

        window.show_image(&frame)?;
    }
}

fn preprocess_image(frame: &Mat) -> Result<Mat> {
    let gray = convert_to_grayscale(frame)?;
    let reduced = reduce_image_size(&gray, SCALE_FACTOR)?;
    equalize_image(&reduced)
}

fn convert_to_grayscale(frame: &Mat) -> Result<Mat> {
    let mut gray = Mat::default()?;
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn reduce_image_size(gray: &Mat, factor: f64) -> Result<Mat> {
    // Destination size is determined by scaling `factor`, not by target size.
    const SIZE_AUTO: Size = Size {
        width: 0,
        height: 0,
    };
    let mut reduced = Mat::default()?;
    imgproc::resize(
        gray,
        &mut reduced,
        SIZE_AUTO,
        factor, // fx
        factor, // fy
        imgproc::INTER_LINEAR,
    )?;
    Ok(reduced)
}

fn equalize_image(reduced: &Mat) -> Result<Mat> {
    let mut equalized = Mat::default()?;
    imgproc::equalize_hist(reduced, &mut equalized)?;
    Ok(equalized)
}

fn detect_faces(
    classifier: &mut objdetect::CascadeClassifier,
    image: Mat,
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
        &image,
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
