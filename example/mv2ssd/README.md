# MobileNetV2 SSD Camera Demo

This guide shows how to compile and deploy the PyTorch model using the MERA compiler.
It provides an example of a fully functional camera demo that uses the deployed model from Python.

## Compile the PyTorch model

### The model
An already quantized and calibrated PyTorch model is provided.
To inspect this model before compiling we could, for example, load it with a network visualizer, for example, Netron.

1. Go to the URL: https://netron.app/
2. Click on "Open Model..." button.
3. Select the file `mv2ssd_640x480.pt` found in the same directory where this readme file is.
4. You can then inspect the model.

Please note that this model has one input and two outputs. One output represents the boxes coordinates and the other output the scores for each class.

### Compile and deploy

To compile and generate shared libraries that contains the deployed model we should run the python script `compile_model.py`:

```bash
python compile_model.py
```
this script creates a new directory named `mv2ssd` in the same directory where `compile_model.py` is located.
It contains the three files necesary to run the model from the either the TVM C++ runtime or the TVM Python runtime.
Given that this demo is written in Python we will use the TVM's Python runtime. For convenience and to show and example of how to load a deployed model from Python a small utility class is provided, this can be found under the `ip_runtime` directory. We will use this class as a helper in the main demo script `ssd_camera_demo.py`.

## Camera demo

The script `ssd_camera_demo.py` shows how to use the previously deployed model in a live camera demo.
The demo script already expects the deployed model `mv2ssd` under the same directory where the camera demo script is located.

To run the demo:

```
python ssd_camera_demo.py 
```

