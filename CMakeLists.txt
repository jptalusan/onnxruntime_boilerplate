cmake_minimum_required(VERSION 3.15)
project(OnnxInfer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Using the brew installation
# # Detect Homebrew prefix automatically
# execute_process(
#     COMMAND brew --prefix onnxruntime
#     OUTPUT_VARIABLE ONNXRUNTIME_INSTALL_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# Use custom-built ONNX Runtime (installed locally)
set(ONNXRUNTIME_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_install/usr/local")

# ORT include and library
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_INSTALL_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.dylib")

add_executable(infer src/infer.cpp)

target_include_directories(infer PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(infer PRIVATE ${ONNXRUNTIME_LIB})

# Make sure the dynamic library can be found at runtime
set_target_properties(infer PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_INSTALL_DIR}/lib"
)