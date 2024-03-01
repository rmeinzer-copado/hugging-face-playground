# https://huggingface.co/docs/transformers/main/en/model_memory_anatomy#load-model

# Import necessary libraries
import numpy as np
from datasets import Dataset
from pynvml import *

# Create dummy data
seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

# Define the print_gpu_utilization function
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# Define the print_summary function
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# Verify initial GPU memory utilization
# ToDo - bailed for  now on this as I need to setup NVIDIA Drivers that was taking too long
print_gpu_utilization()
