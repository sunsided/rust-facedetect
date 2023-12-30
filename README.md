# OpenCV Face Detection in Rust

An experiment with [opencv-rust](https://github.com/twistedfall/opencv-rust) and basically not much more than the multi-scale face detection demo
(using a frontal-face [Haar cascade](https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html)) on the first video capture device that can be found. 

To run it, execute

```bash
cargo run
```

To exit, press `ESC`.

## Startup errors

If you get errors such as 

```
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (1100) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (2075) handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Internal data stream error.
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (651) startPipeline OpenCV | GStreamer warning: unable to start pipeline
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (1257) setProperty OpenCV | GStreamer warning: no pipeline
thread 'main' panicked at src/main.rs:41:9:
Unable to open default camera!
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (616) isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
```

then the default camera parameters are incorrect for your device.
To select different parameters, use the `CAPTURE_WIDTH`, `CAPTURE_HEIGHT` and `CAPTURE_DEVICE` environment variables:

```shell
CAPTURE_WIDTH=848 CAPTURE_HEIGHT=480 CAPTURE_DEVICE=1 cargo run --release
```
