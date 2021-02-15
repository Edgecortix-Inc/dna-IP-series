import os
import numpy as np

from tvm import runtime
from tvm.contrib import graph_runtime


def run_on_board(deployment_dir, input_name, input_data):
    lib_path = os.path.join(deployment_dir, "deploy.so")
    with open(os.path.join(deployment_dir, "deploy.json"), "r") as f:
        json_data = f.read()
    with open(os.path.join(deployment_dir, "deploy.params"), "rb") as f:
        parameters_data = f.read()
    lib = runtime.load_module(lib_path)
    ctx = runtime.cpu()
    rt_mod = graph_runtime.create(json_data, lib, ctx)
    rt_mod.load_params(parameters_data)
    rt_mod.set_input(input_name, input_data)
    rt_mod.run()
    ec_result_0 = rt_mod.get_output(0).asnumpy()
    ec_latency = rt_mod.get_elapsed_latency()
    return ec_result_0, ec_latency


# Set deployment directory, input name, input shape and output shape
deployment_dir = '/opt/edgecortix/resnet50_deploy'
input_name = "input0"
input_shape = ((1, 224, 224, 3))
output_shape_0 = ((1, 1000))

# Load the reference data
input_data = np.fromfile(os.path.join(deployment_dir, "input0.bin"), dtype=np.float32).reshape(input_shape)
reference_result0 = np.fromfile(os.path.join(deployment_dir, "ref_result_0.bin"), dtype=np.float32).reshape(output_shape_0)

# Run on hardware
ec_result_0, ec_latency = run_on_board(deployment_dir, input_name, input_data)
print("Total latency:", ec_latency, " microseconds")
print("Results 0 match: ", np.allclose(reference_result0, ec_result_0))

