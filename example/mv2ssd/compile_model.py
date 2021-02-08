import os
import torch
import numpy as np

import tvm
from tvm import relay
from tvm.relay import mera


def export_module(inp_shape, script_module, output_dir, config):
    input_name = "input0"
    input_layout = "NHWC"
    input_shapes = [(input_name, inp_shape)]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes, layout=input_layout)

    with mera.build_config(target="IP", **config):
        mera.build(mod, params, output_dir=output_dir,
                 host_arch="x86", layout=input_layout)


out_dir = "mv2ssd"
config = {"arch": 200}
inp_shape = (1, 480, 640, 3)
model = torch.jit.load("../model_zoo/mv2ssd_640x480.pt")
export_module(inp_shape, model, out_dir, config)

