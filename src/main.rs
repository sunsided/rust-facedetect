extern crate opencv;
use opencv::core::{Rect, Scalar, Size};
use opencv::{highgui, imgproc, objdetect, prelude::*, types, videoio};

const SCALE_FACTOR: f64 = 0.25f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

fn run() -> opencv::Result<()> {
    let xml = "haarcascade_frontalface_alt.xml";
    let mut classifier = objdetect::CascadeClassifier::new(&xml)?;

    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    capture.set(videoio::CAP_PROP_FRAME_WIDTH, 800f64)?;
    capture.set(videoio::CAP_PROP_FRAME_HEIGHT, 600f64)?;

    let opened = videoio::VideoCapture::is_opened(&capture)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let window_name = "OpenCV Face Detection in Rust";
    highgui::named_window(
        window_name,
        highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO,
    )?;

    highgui::resize_window(window_name, 800, 600)?;

    loop {
        if let Ok(27) = highgui::wait_key(10) {
            break;
        }

        if !capture.grab()? {
            continue;
        }

        let mut frame = Mat::default()?;
        capture.retrieve(&mut frame, 0)?;

        let mut gray = Mat::default()?;
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

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

        let mut faces = types::VectorOfRect::new();
        classifier.detect_multi_scale(
            &equalized,
            &mut faces,
            1.1,
            2,
            0,
            Size {
                width: 30,
                height: 30,
            },
            Size {
                width: 0,
                height: 0,
            },
        )?;

        for face in faces {
            println!("face {:?}", face);
            let scaled_face = Rect {
                x: face.x * SCALE_FACTOR_INV,
                y: face.y * SCALE_FACTOR_INV,
                width: face.width * SCALE_FACTOR_INV,
                height: face.height * SCALE_FACTOR_INV,
            };
            imgproc::rectangle(
                &mut frame,
                scaled_face,
                Scalar::new(0f64, 0f64, 255f64, -1f64),
                2,
                8,
                0,
            )?;
        }

        highgui::imshow(window_name, &frame)?;
    }

    highgui::destroy_window(window_name)?;
    capture.release()?;
    Ok(())
}

fn main() {
    run().unwrap()
}
