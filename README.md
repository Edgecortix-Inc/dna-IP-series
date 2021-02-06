
# Deploying a CNN model with EdgeCortix Dynamic Neural Accelerator® (DNA-F100/F200) and the MERA™ compiler

This guide assumes that:
- The DNN model has been created with the PyTorch DL framework.
- Deployment target is a Xilinx ALVEO U50 PCI-E board.
- Running in EdgeCortix docker environment (Nimbix or on-premise).

**NOTE**: We can run MERA compiler under *release* or *profiling* (fine-grained performance assessment) modes. Here we only consider the *release* mode version. 

Once inside the docker environment, any shell that is launched should already have a Python virtual environment activated. This means that we have access to EdgeCortix's MERA compiler stack and all its dependencies. We are ready for compiling an DNN model for our target hardware.

## Create and compile a DNN model with MERA
In general, the compilation process involve the following steps:

1. Create a DNN model (DNA does not need quantization aware training, however it should be possible to quantize the model post-training to INT8 bit).
2. Quantize and JIT trace the model (pytorch quantization example)[https://pytorch.org/docs/stable/quantization.html].
3. Run the model on the MERA interpreter to get reference results.
4. Import the traced model and compile with MERA.
5. Create a shared library that contains the deployable binaries.

### Step 1 - Create a quantizable model

For simplicity we will use pre-trained model provided by the `torchvision` (library)[https://github.com/pytorch/vision/tree/master/torchvision/models/quantization] which is commonly installed together with the `pytorch` (package)[https://pytorch.org/]. These are already pre-installed inside our virtual environment in docker.

First, import all the necessary packages:

```python
import os
import torch
import numpy as np
import tvm
from tvm import relay
from tvm.relay import mera
```

Next, import the pre-trained model:

```python
from torchvision.models.quantization import resnet as qresnet
model = qresnet.resnet50(pretrained=True).eval()
```

### Step 2 - Quantize and JIT trace the model

To deploy a model, we will use the built-in post-training quantization of Pytorch. Tracing the model is also necessary because the TVM PyTorch frontend expects a traced model as an input.

```python
# ResNet-50 Pytorch model expects an input image with layout NCHW and size 224x224. We create a random input tensor for our test
inp = torch.rand((1, 3, 224, 224))

model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
model(inp)
torch.quantization.convert(model, inplace=True)

with torch.no_grad():
    script_module = torch.jit.trace(model, inp).eval()
```

### Step 3 - Run the model on MERA interpreter

This step will generate reference input and output data that will be used later to verify that the deployment succeeded. As such, this step is a recommended sanity check. We run the same input through Edgecortix's MERA interpreter and save each reference result.

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

with mera.build_config(target="InterpreterHw"):
    int_result = mera.test_util.run_mera_backend(script_module, nhwc_inp, layout="NHWC")
    res_idx = 0
    for res in int_result:
        nhwc_res = res
        nhwc_res.flatten().astype(np.float32).tofile(
            os.path.join(output_dir, "ref_result_" + str(res_idx) + ".bin"))
        res_idx += 1
```

### Step 4 - Import the traced model from step 2

```python
input_shapes = [(input_name, nhwc_inp.shape)]
mod, params = relay.frontend.from_pytorch(script_module,input_shapes,layout=input_layout)
```

### Step 5 - Create the shared library

At this stage, we are ready to deploy to **real hardware** (target="IP") or the MERA **simulator** (target="Simulator") The `arch` parameter in the script should be chosen depending on which release of the DNA IP is being used. e.g. the value should be `100` for the DNA-F100 release, `200` for the DNA-F200 release.

```python
config = {
    "arch": 100,
}
with mera.build_config(target="IP", **config):
    mera.build(mod, params, output_dir=output_dir, host_arch="x86", layout=input_layout)
```

After running the python script, an output similar to the following should be seen when target is real hardware:

```
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /home/nimbix/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:01<00:00, 52.6MB/s]
/opt/edgecortix/pyenv/lib/python3.6/site-packages/torch/quantization/observer.py:121: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  reduce_range will be deprecated in a future release of PyTorch."
...100%, 0.02 MB, 137 KB/s, 0 seconds passed
Cannot find config for target=llvm -keys=cpu -mcpu=core-avx2, workload=('dense_nopack.x86', ('TENSOR', (1, 2048), 'int16'), ('TENSOR', (1000, 2048), 'int16'), None, 'int32'). A fallback configuration is used, which may bring great performance regression.
Elapsed 20.552747 seconds
Simulator total latency: 0
```

This script will create a new directory named `resnet50_deploy` in the current directory. This directory contains a shared library that can be used to run the model by using the TVM runtime along with reference data that will be used to validate the deployment. An example of the files found on this newly-created directory are:

```
deploy.json
deploy.params
deploy.so
input0.bin
ref_result_0.bin
```

Now we are ready to run this model on the real hardware. For convenience, our docker container comes with a simple C++ application (for CNN classification) provided under `/opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp`.

To build this application, we should run:

```bash
cd /opt/edgecortix/private-tvm/apps/mera_cpp/
mkdir build
cd build
cmake ..
make
```

This will create an executable file named `inference`. To run this application, we only need to provide the deployment directory. Assuming that the deployment directory is under `/opt/edgecortix/resnet50_deploy`, we can now run the model:

```bash
./inference /opt/edgecortix/resnet50_deploy/
```

This will run the MERA compiled model on the FPGA board, as well as compare the reference results against the the ones we saved previously during the compilation of the model. The successful output will be similar to the following snippet (the time mentioned is the actual end-to-end latency for batch size 1):

```
[22:53:30] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:63: Loading json data...
[22:53:30] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:69: Loading runtime module...
Found Platform
Platform Name: Xilinx
INFO: Reading /opt/edgecortix/dna.xclbin
Loading: '/opt/edgecortix/dna.xclbin'
[  info  ] 657   , DRM session CD1A42C64A2EEAA8 created.
[22:53:32] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:74: Loading parameters...
[22:53:32] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:82: Loading input...
[22:53:32] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:114: Warming up...
[22:53:32] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:120: Running 100 times...
[22:53:33] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:127: Took 7.96 msec.
[22:53:33] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:47: max abs diff: 1.17549e-38
[22:53:33] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:48: mean abs diff: 0
[22:53:33] /opt/edgecortix/private-tvm/apps/mera_cpp/inference.cpp:49: correct ratio: 1
[  info  ] 657   , DRM session CD1A42C64A2EEAA8 stopped.
```

## Compile other example DNN models
The `example` folder in the repository includes ready-to-run python scripts for some example deep neural networks, including Resnet-50 which was discussed above. These scripts assume the F100 variation of the DNA IP. For other IP versions, the value of the `arch` parameter should be updated accordingly. After this, the script can be compiler and run as is, based on the instructions in the previous section.
