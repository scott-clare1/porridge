# Step 1: Build the Rust project
FROM rust:latest as builder

WORKDIR /usr/src/
RUN rustup target add x86_64-unknown-linux-musl

RUN USER=root cargo new --bin porridge
WORKDIR /usr/src/porridge

# copy over your manifests
COPY Cargo.toml Cargo.lock ./

# this build step will cache your dependencies
RUN cargo build --release
RUN rm src/*.rs

# copy your source tree
COPY ./src ./src

# build for release
RUN cargo install --target x86_64-unknown-linux-musl --path .

# our final base
FROM scratch

# copy the build artifact from the build stage
COPY --from=builder /usr/local/cargo/bin/porridge .

EXPOSE 3030

# set the startup command to run your binary
CMD ["./porridge"]
