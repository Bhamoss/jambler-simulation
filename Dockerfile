FROM rustembedded/cross:armv7-unknown-linux-gnueabihf
RUN apt-get update
RUN apt-get install --assume-yes gcc pkg-config openssl libasound2-dev cmake build-essential python3 libfreetype6-dev libexpat1-dev libxcb-composite0-dev libssl-dev libx11-dev libfontconfig1-dev