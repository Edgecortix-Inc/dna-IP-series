#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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

