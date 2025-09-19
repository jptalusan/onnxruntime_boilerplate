cmake_minimum_required(VERSION 3.15)
project(OnnxInferenceProjects CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCE_FILES "src/*.cpp")

# Detect Homebrew prefix automatically
# execute_process(
# COMMAND brew --prefix onnxruntime
# OUTPUT_VARIABLE ONNXRUNTIME_INSTALL_DIR
# OUTPUT_STRIP_TRAILING_WHITESPACE
# )

find_package(nlohmann_json REQUIRED)

set(ONNXRUNTIME_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_install/usr/local")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_INSTALL_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.dylib")

foreach(SOURCE_FILE IN LISTS SOURCE_FILES)
# Get the file name without the path and extension (e.g., "onnx_inference")
get_filename_component(EXECUTABLE_NAME "${SOURCE_FILE}" NAME_WE)

# Add an executable for the current source file
add_executable(${EXECUTABLE_NAME} "${SOURCE_FILE}")

# Link the required libraries and include directories to the new executable
target_include_directories(${EXECUTABLE_NAME} PRIVATE 
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE 
    ${ONNXRUNTIME_LIB}
    nlohmann_json::nlohmann_json
)

# Ensure the dynamic library can be found at runtime
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_INSTALL_DIR}/lib"
)

endforeach()