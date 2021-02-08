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
import numpy as np

from tvm import runtime
from tvm.contrib import graph_runtime


class IPRuntime(object):
    def __init__(self):
        self.rt_mod = None
        self.input_index = 0
        self.num_outputs = 0

    def Setup(self, model_lib_path: str):
        with open(os.path.join(model_lib_path, "deploy.json"), "r") as f:
            json_data = f.read()

        with open(os.path.join(model_lib_path, "deploy.params"), "rb") as f:
            parameters_data = f.read()
        
        lib = runtime.load_module(os.path.join(model_lib_path, "deploy.so"))
        self.rt_mod = graph_runtime.create(json_data, lib, runtime.cpu())
        self.rt_mod.load_params(parameters_data)
        self.num_outputs = self.rt_mod.get_num_outputs()

    def Run(self, input_data):
        self.rt_mod.set_input(self.input_index, input_data)
        self.rt_mod.run()

        outputs = []
        for idx in range(self.num_outputs):
            outputs.append(np.asarray(self.rt_mod.get_output(idx).asnumpy()))
        return outputs
