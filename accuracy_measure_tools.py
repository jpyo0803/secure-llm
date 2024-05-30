import torch
import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Choose a GPU to run the benchmark

DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
assert DEVICE is not None, "CUDA_VISIBLE_DEVICES is not set."
# Must choose a clock speed that's supported on your device.
CLOCK_SPEED = 1350


def set_clock_speed():
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    process = subprocess.Popen(
        "nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(
        f"sudo nvidia-smi -pm ENABLED -i {DEVICE}",      shell=True)
    process = subprocess.run(
        f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True)
    print("Clock speed set to: ", CLOCK_SPEED)


def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)
    print("Clock speed reset to default.")


tensor_for_flush = torch.empty(
    int(10 * (1024 ** 2)), dtype=torch.int8, device=torch.device("cuda:0"))  # 10 MB


def flush_cache():
    tensor_for_flush.zero_()
