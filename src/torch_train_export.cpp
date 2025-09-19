#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <algorithm>
#include <onnxruntime/onnxruntime_cxx_api.h>
// NOTE: You will need to install a JSON parsing library for C++, such as nlohmann/json.
// https://github.com/nlohmann/json
// A common way to install it is via vcpkg or by simply including the header file.
// Compilation command with nlohmann/json:
// g++ onnx_inference.cpp -o onnx_inference -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -I/path/to/nlohmann/json/include

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Global variables to store the loaded mapping
std::vector<std::string> featureNames;
std::map<std::string, size_t> categoricalMapping;

// Function to load the feature mapping from a JSON file
bool loadFeatureMapping(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open feature mapping file " << filepath << std::endl;
        return false;
    }

    try {
        json mapping_data;
        file >> mapping_data;
        file.close();

        // Load the full list of feature names
        featureNames = mapping_data["features"].get<std::vector<std::string>>();

        // Create a map for the categorical features for quick lookup
        std::vector<std::string> cat_features = mapping_data["categorical_mapping"].get<std::vector<std::string>>();
        for (size_t i = 0; i < cat_features.size(); ++i) {
            categoricalMapping[cat_features[i]] = i;
        }
        std::cout << "Feature mapping loaded successfully." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to prepare input data dynamically using the loaded mapping
std::vector<float> prepareInputData(const std::string& city, float age) {
    std::vector<float> features(featureNames.size(), 0.0f);

    // Find the index for the one-hot encoded city
    std::string city_feature_name = "city_" + city;
    auto it = categoricalMapping.find(city_feature_name);
    if (it != categoricalMapping.end()) {
        features[it->second] = 1.0f;
    } else {
        std::cerr << "Warning: Unknown city '" << city << "'. One-hot encoded feature will be all zeros." << std::endl;
    }

    // Find the index for the numerical age feature and set its value
    auto age_it = std::find(featureNames.begin(), featureNames.end(), "age");
    if (age_it != featureNames.end()) {
        features[std::distance(featureNames.begin(), age_it)] = age;
    } else {
        std::cerr << "Error: 'age' feature not found in mapping!" << std::endl;
    }

    return features;
}

int main() {
    // --- 1. Load the feature mapping JSON ---
    if (!loadFeatureMapping("../data/features_mapping.json")) {
        return 1;
    }

    // --- 2. Read/Simulate Dummy Data ---
    std::vector<std::pair<std::string, float>> new_data = {
        {"London", 35.0f},
        {"New York", 42.0f},
        {"Berlin", 30.0f}
    };

    // --- 3. Load the ONNX Model ---
    std::cout << "\nLoading ONNX model from model.onnx..." << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../data/model.onnx", session_options);
    std::cout << "Model loaded successfully." << std::endl;

    // --- 4. Prepare Input and Perform Inference ---
    for (const auto& entry : new_data) {
        std::string city = entry.first;
        float age = entry.second;

        std::vector<float> input_data = prepareInputData(city, age);

        // --- UPDATED SECTION: Getting input/output names and shape ---
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        
        // Print the inferred input shape for verification
        std::cout << "Inferred input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Use the C-style string from the allocated pointers
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();
        std::vector<const char*> input_names_ptr = {input_name};
        std::vector<const char*> output_names_ptr = {output_name};

        // Create the input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        // Run inference
        std::cout << "\nRunning inference for: City=" << city << ", Age=" << age << std::endl;
        try {
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_ptr.data(), &input_tensor, 1, output_names_ptr.data(), 1);

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            float prediction = output_data[0];

            std::cout << "Prediction (sigmoid output): " << prediction << std::endl;
            if (prediction > 0.5) {
                std::cout << "Inferred class: Class 1" << std::endl;
            } else {
                std::cout << "Inferred class: Class 0" << std::endl;
            }

        } catch (const Ort::Exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
