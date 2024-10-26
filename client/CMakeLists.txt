cmake_minimum_required(VERSION 3.17)

project(cc-clients LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")

#
# Options
#
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" ON)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" ON)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)
option(TRITON_USE_THIRD_PARTY "Use local version of third party libraries" OFF)
option(TRITON_KEEP_TYPEINFO "Keep typeinfo symbols by disabling ldscript" OFF)
option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)

if(TRITON_ENABLE_CC_GRPC)
  set(TRITON_COMMON_ENABLE_PROTOBUF ON)
  set(TRITON_COMMON_ENABLE_GRPC ON)
endif() # TRITON_ENABLE_CC_GRPC

if(NOT TRITON_ENABLE_CC_HTTP AND NOT TRITON_ENABLE_EXAMPLES)
  set(TRITON_COMMON_ENABLE_JSON OFF)
endif()

if(TRITON_ENABLE_TESTS)
  FetchContent_MakeAvailable(googletest)
endif()
FetchContent_MakeAvailable(repo-common)

if(TRITON_ENABLE_TESTS)
  include_directories(
    ${repo-common_SOURCE_DIR}/include
  )
endif() # TRITON_ENABLE_TESTS

#
# CUDA
#
if(TRITON_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# libcurl
#
if(TRITON_ENABLE_CC_HTTP)
  find_package(CURL REQUIRED)
  message(STATUS "Using curl ${CURL_VERSION}")
endif() # TRITON_ENABLE_CC_HTTP

#
# Protobuf
#
if(TRITON_ENABLE_CC_GRPC)
  set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")
  include_directories(${Protobuf_INCLUDE_DIRS})
endif() # TRITON_ENABLE_CC_GRPC

#
# GRPC
#
if(TRITON_ENABLE_CC_GRPC)
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")
  include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
endif() # TRITON_ENABLE_CC_GRPC

# Include the ExternalProject module
include(ExternalProject)

# Declare the external content
FetchContent_Declare(
  triton_client
  GIT_REPOSITORY https://github.com/triton-inference-server/client.git
  GIT_TAG main
)

# Only download the content, don't configure or build it
FetchContent_GetProperties(triton_client)
if(NOT triton_client_POPULATED)
  FetchContent_Populate(triton_client)
endif()

# Get the source directory
FetchContent_GetProperties(triton_client SOURCE_DIR TRITON_CLIENT_SOURCE_DIR)

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)
    add_subdirectory(${TRITON_CLIENT_SOURCE_DIR}/src/c++/library)
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC

add_executable(main_test src/main.cpp)
target_link_libraries(
  main_test
  PRIVATE
    httpclient_static
)