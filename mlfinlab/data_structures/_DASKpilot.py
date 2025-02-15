import cupy as cp
import dask.dataframe as dd
from dask.distributed import Client
import cudf
import dask_cudf
import numpy as np
import dask.array as da
import time

# https://chat.deepseek.com/a/chat/s/1d80fb82-0855-4095-ac89-373f26146c91

# Function to compute sum with Dask
def compute_with_dask(data):
    # Start the timer
    start_time = time.time()
    
    # Compute the sum
    result = data.sum().compute()
    
    # Stop the timer
    end_time = time.time()
    
    print(f"Sum with Dask: {result}")
    print(f"Time taken with Dask: {end_time - start_time:.4f} seconds")


def compute_with_daskgpu(data):
    # Convert Dask array to CuPy array (GPU)
    data = data.map_blocks(cp.asarray)

    # Start the timer
    start_time = time.time()

    # Compute the sum on GPU
    result = data.sum().compute()

    # Stop the timer
    end_time = time.time()

    print(f"Sum with Dask (GPU): {result}")
    print(f"Time taken with Dask (GPU): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    data = da.random.random(1e11, chunks=1e6) 

    print("\nRunning computation with Dask...")
    compute_with_dask(data) # for (1e11, chunks=1e6), 71s

    print("\nRunning computation with DaskGPU...")
    compute_with_daskgpu(data) # for (1e11, chunks=1e6), s
