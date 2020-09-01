# OpenCV Face Detection in Rust

An experiment with [opencv-rust](https://github.com/twistedfall/opencv-rust) and
basically not much more than the multi-scale face detection demo
(using a frontal-face [Haar cascade](https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html))
on the first video capture device that can be found. 

To run it, execute

```bash
cargo run
```

To exit, press `ESC`.

## Build environment

This project was built on Ubuntu 20.04 with OpenCV 4.2.0 using Rust 1.46.0, Clang/LLVM 10 and
`opencv-rust` version `0.45.1`. I couldn't get the project to build with a default
configuration (i.e. `opencv = "0.45"`), but eventually got a combination of features that worked:

- `opencv-4`
- `buildtime-bindgen`
- `clang-runtime`

That is,

```toml
[dependencies]
opencv = {version = "0.45", features = ["opencv-4", "buildtime-bindgen", "clang-runtime"]}
```

A [Dockerfile](Dockerfile) is provided as an example build environment for the above setup.
