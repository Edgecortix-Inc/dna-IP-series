import os
import torch
import numpy as np
import tvm
from tvm import relay
from tvm.relay import mera

from torchvision.models.quantization import mobilenet as qmobilenet
model = qmobilenet.mobilenet_v2(pretrained=True).eval()

# Mobilenet V2 PyTorch model expects an input image with layout NCHW and size 224x224
inp = torch.rand((1, 3, 224, 224))
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
model(inp)
torch.quantization.convert(model, inplace=True)
with torch.no_grad():
    script_module = torch.jit.trace(model, inp).eval()
    
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
output_dir = "mobilenetv2_deploy"
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

input_shapes = [(input_name, nhwc_inp.shape)]
mod, params = relay.frontend.from_pytorch(script_module,input_shapes,layout=input_layout)

config = {
    "arch": 100,
}

with mera.build_config(target="IP", **config):
    mera.build(mod, params, output_dir=output_dir, host_arch="x86", layout=input_layout)
