# Step 1: Build the Rust project
FROM rust:latest as builder

RUN USER=root cargo new --bin porridge
WORKDIR /porridge

# copy over your manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

# this build step will cache your dependencies
RUN cargo build --release
RUN rm src/*.rs

# copy your source tree
COPY ./src ./src

# build for release
RUN rm ./target/release/deps/porridge*
RUN cargo build --release

# our final base
FROM rust:latest

# copy the build artifact from the build stage
COPY --from=builder /porridge/target/release/porridge .

EXPOSE 3030

# set the startup command to run your binary
CMD ["./porridge"]
