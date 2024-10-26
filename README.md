# Send memory buffer through triton client

Convert tensor array to buffer of type `uint64`, send through the `TRITON` inference server and send back. Brief instructions on how to build in each folder. 

## Docker

Docker image is at `docker.io/milescb/triton-server:latest`

## Build the backend

Go to the backend directory and make a build and install directory:

```
mkdir build && install
cd build
```

Then build

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_INSTALL_PREFIX=../install/

cmake --build . --target install -- -j20
```

## Run the standalone test

Open two terminals. In one, start the server (after building) with

```
tritonserver --model-repository=model_repo
```

Then run the `python` client (`c++` version still in dev.)