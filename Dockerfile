# This Dockerfile provides an example of a build environment for Ubuntu 20.04, OpenCV 4.2 and opencv-rust 0.45.
# Note that the resulting image is neither optimized for size nor meant to run the produced binary.

FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install base environment.
RUN apt-get update
RUN apt-get install -y \
    curl \
    build-essential \
    git

# Install Rust 1.46.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh && sh /tmp/rustup.sh -y

# Install dependencies for opencv-rust.
RUN apt-get install -y \
    clang \
    libopencv-dev \
    libclang-dev

# Building the application.
WORKDIR /usr/src/rust-facedetect
RUN $HOME/.cargo/bin/rustup override set 1.46.0

COPY Cargo.* ./
COPY src/ ./src

RUN $HOME/.cargo/bin/cargo build --release

# Application built as ./target/release/facedetect
