
# How to deploy ResNet-50 model on Alveo U50 FPGA

In this guide we will assume that:
- our DL model has been created with the PyTorch DL framework.
- we will deploy on a host with a U50 FPGA PCIe board
- we have access to an EdgeCortix docker environment

Once in the docker environment, the bash terminal shows that a Python virtual environment is already activated, this means we have access to TVM along with all its dependencies and we are ready for compiling a DL model for our target.

In general, the compilation process involve several steps:

1. Create a quantizable model
2. Quantize and trace the model
3. Run the model on the Edgecortix interpreter to get reference results
4. Import the traced model into TVM
5. Create a shared library that contains our deployed model

## Step 1 - Create a quantizable model

For simplicity we will use an already created and trained model provided
by the `torchvision` package which is commonly installed together with the `torch` package. These are already installed on the virtual environment.

First, import all the necessary packages:

```python
import os
import torch
import numpy as np
import tvm
from tvm import relay
from tvm.relay import ec
```

Then import the model:

```python
from torchvision.models.quantization import resnet as qresnet
model = qresnet.resnet50(pretrained=True).eval()
```

## Step 2 - Quantize and trace the model

To deploy a model we should use the built-in post-training quantization implementation of Pytorch. Tracing a model is also necessary because the TVM PyTorch front-end expects to a traced model as an input.

```python
# ResNet-50 Pytorch model expects an input image with layout NCHW and size 224x224
inp = torch.rand((1, 3, 224, 224))
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
model(inp)
torch.quantization.convert(model, inplace=True)
with torch.no_grad():
    script_module = torch.jit.trace(model, inp).eval()
```

## Step 3 - Run the model on Edgecortix interpreter

This step will generate reference input and output data that will be used later to verify that the deployment succeeded. We run the same input through the Edgecortix interpreter and save each reference result.

```python
def nchw_to_nhwc(arr):
    if len(arr.shape) != 4:
        return arr
    N, C, H, W = arr.shape
    ret = np.zeros((N, H, W, C), dtype=arr.dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    ret[n, h, w, c] = arr[n, c, h, w]
    return ret

# Save reference input and run the model on the interpreter
output_dir = "resnet50_deploy"
input_name = "input0"
input_layout = "NHWC"
nhwc_inp = nchw_to_nhwc(inp.numpy())
os.makedirs(output_dir, exist_ok=True)
nhwc_inp.flatten().astype(np.float32).tofile(os.path.join(output_dir, input_name+".bin"))

with ec.build_config(target="InterpreterHw"):
    int_result = ec.test_util.run_ec_backend(script_module, nhwc_inp, layout="NHWC")
    res_idx = 0
    for res in int_result:
        nhwc_res = res
        nhwc_res.flatten().astype(np.float32).tofile(
            os.path.join(output_dir, "ref_result_" + str(res_idx) + ".bin"))
        res_idx += 1
```

## Step 4 - Import the traced model

```python
input_shapes = [(input_name, nhwc_inp.shape)]
mod, params = relay.frontend.from_pytorch(script_module,input_shapes,layout=input_layout)
```

## Step 5 - Create the shared library

At this stage we are ready for deployment. It is important to populate a Python dictionary with some specific entries provided by Edgecortix.

```python
config = {
# provided by Edgecortix
}
with ec.build_config(target="IP", **config):
    ec.build(mod, params, output_dir=output_dir, host_arch="x86", layout=input_layout)
```

At this point a new directory named `resnet50_deploy` should exist in the current directory. This directory contains a shared library that can be used to run the model by using the TVM runtime along with reference data that will be used to validate the deployment. An example of the files found on this newly created directory are:

```
deploy.json
deploy.params
deploy.so
input0.bin
ref_result_0.bin
```

Now we are ready to run this model on the real hardware. For convenience, a simple C++ application is provided under `/opt/edgecortix/private-tvm/apps/ec_cpp/inference.cpp`.

To build this application we should run:

```bash
cd /opt/edgecortix/private-tvm/apps/ec_cpp/
mkdir build
cd build
cmake ..
make
```

this will create an executable file named `inference`. This application is able to accept the deployment directory created during the previous steps.

It is important to note that the bitstream file with extension `.xclbin` , which will also be provided by Edgecortix, should be placed on the same directory where the `inference` application is invoked.

To run this application we should only provide the deployment directory. Assuming that the deployment directory is under `/opt/edgecortix/resnet50-deploy` and the bitstream has been placed under the current directory where we will run `inference` application we can now run the model:

```bash
./inference /opt/edgecortix/resnet50-deploy/
```

This will run the model on the U50 board as well as compare the results against the the ones we saved previously during the compilation of the model.
