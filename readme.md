# ONNX/ONNXRuntime sandbox
I am trying to understand and create ML models (from sklearn or torch) and call them for, inference, within a c++ pipeline. 
I am on Python 3.12 running on M3 Pro 15.6.

## Easy Setup
You can start using `onnxruntime` using the brew installation however, there might some `schema error` warnings that will pop up during use but those can be ignored.
```bash
pip install torch onnx onnxruntime skl2onnx
pip install --upgrade onnx onnxscript
brew install onnxruntime
```

## Building from Source
This is preferred to have more control and it will help remove the `schema error` warnings.
```bash
# Unlink prerequisites (temporarily)
brew unlink abseil
brew unlink re2
brew unlink protobuf
# unlink or even uninstall these two.
brew unlink onnx
brew unlink onnxruntime
brew install pkg-config

# Build from source
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --use_vcpkg
# for the repo, i installed it to where the repo was cloned.
# git/onnxruntime_boilerplate
make install DESTDIR=[DESTINATION_DIR]
# example: make install DESTDIR=/fire_simulator/external
```

### Building the code
```bash
cd onnxruntime_boilerplate
mkdir build
cd build
cmake ..
make -j4
./infer
# Output
Input name: input
Input shape: -1 3 
Prediction: 217.607
Predicted y = 7.03005
```

## Models
This works for both `scikit-learn` and `pytorch` models.  
Generate the models using `python scripts/generate_models.py` it should be saved in `data` folder.


# References
* [Deploying PyTorch Model into a C++ Application Using ONNX Runtime](https://medium.com/@freshtechyy/deploying-pytorch-model-into-a-c-application-using-onnx-runtime-f9967406564b)
* [ONNX Runtime C++ Inference](https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/)
