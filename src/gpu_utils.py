"""
GPU Utilities for Radio Base Station Analysis

This module provides utilities for managing GPU acceleration across
different parts of the codebase. It centralizes GPU detection and
configuration, and provides helper functions for common operations.
"""

import os
import warnings
import numpy as np
import time

# Global flag to indicate if GPU acceleration is available
GPU_AVAILABLE = False
GPU_DEVICE = None
USE_GPU = os.environ.get('USE_GPU', '').lower() == 'true'

# Check if PyTorch is available with GPU support
try:
    import torch
    TORCH_AVAILABLE = True
    
    if torch.cuda.is_available() and USE_GPU:
        GPU_AVAILABLE = True
        GPU_DEVICE = torch.device('cuda')
        print(f"PyTorch GPU acceleration available: {torch.cuda.get_device_name(0)}")
    else:
        GPU_DEVICE = torch.device('cpu')
        if USE_GPU and not torch.cuda.is_available():
            warnings.warn("PyTorch GPU acceleration requested but not available. Using CPU.")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. PyTorch-based GPU acceleration unavailable.")

# Check if CuPy is available for NumPy-like operations on GPU
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    if USE_GPU:
        if GPU_AVAILABLE:
            print("CuPy acceleration available for NumPy-like operations.")
        else:
            # Only set GPU_AVAILABLE if not already set by PyTorch
            GPU_AVAILABLE = True
            print(f"CuPy GPU acceleration available: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    else:
        print("CuPy found but GPU acceleration not enabled.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. NumPy-like GPU acceleration unavailable.")

# Check if cuSpatial is available for spatial operations
try:
    import cuspatial
    CUSPATIAL_AVAILABLE = True
    
    if USE_GPU and (GPU_AVAILABLE or CUPY_AVAILABLE):
        print("cuSpatial acceleration available for spatial operations.")
    else:
        print("cuSpatial found but GPU acceleration not enabled.")
except ImportError:
    CUSPATIAL_AVAILABLE = False
    print("cuSpatial not found. GPU spatial acceleration unavailable.")

def get_array_module(use_gpu=None):
    """
    Returns the appropriate array module (numpy or cupy) based on
    whether GPU acceleration is available and enabled.
    
    Args:
        use_gpu (bool, optional): Override global GPU setting
        
    Returns:
        module: numpy or cupy module
    """
    use_gpu_local = USE_GPU if use_gpu is None else use_gpu
    if use_gpu_local and CUPY_AVAILABLE:
        return cp
    return np

def to_device(data, use_gpu=None):
    """
    Move data to the appropriate device (CPU or GPU)
    
    Args:
        data: Data to move (tensor, array, or compatible object)
        use_gpu (bool, optional): Override global GPU setting
        
    Returns:
        object: Data on the appropriate device
    """
    use_gpu_local = USE_GPU if use_gpu is None else use_gpu
    
    if not use_gpu_local:
        # Move to CPU if available
        if hasattr(data, 'cpu'):
            return data.cpu()
        return data
    
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return data.to(GPU_DEVICE)
    
    if CUPY_AVAILABLE:
        if isinstance(data, np.ndarray):
            try:
                return cp.asarray(data)
            except Exception as e:
                warnings.warn(f"Failed to move NumPy array to GPU: {e}")
                return data
    
    return data

def to_cpu(data):
    """
    Move data to CPU memory
    
    Args:
        data: Data to move (tensor, array, or compatible object)
        
    Returns:
        object: Data on CPU
    """
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    
    if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    
    return data

def time_function(func, *args, repeat=3, **kwargs):
    """
    Time a function's execution with and without GPU acceleration
    
    Args:
        func: Function to time
        *args: Arguments to pass to the function
        repeat: Number of times to repeat the timing
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        dict: Dictionary with CPU and GPU timing results
    """
    results = {}
    
    # Time without GPU
    os.environ['USE_GPU'] = 'false'
    cpu_times = []
    for _ in range(repeat):
        start = time.time()
        func(*args, **kwargs)
        cpu_times.append(time.time() - start)
    results['cpu'] = {
        'mean': np.mean(cpu_times),
        'min': np.min(cpu_times),
        'max': np.max(cpu_times)
    }
    
    # Time with GPU if available
    if GPU_AVAILABLE:
        os.environ['USE_GPU'] = 'true'
        gpu_times = []
        for _ in range(repeat):
            start = time.time()
            func(*args, **kwargs)
            gpu_times.append(time.time() - start)
        results['gpu'] = {
            'mean': np.mean(gpu_times),
            'min': np.min(gpu_times),
            'max': np.max(gpu_times)
        }
        
        # Calculate speedup
        results['speedup'] = results['cpu']['mean'] / results['gpu']['mean']
    
    # Reset environment variable to its original state
    if USE_GPU:
        os.environ['USE_GPU'] = 'true'
    else:
        os.environ['USE_GPU'] = 'false'
    
    return results

def memory_usage():
    """
    Get current GPU memory usage if available
    
    Returns:
        dict: Dictionary with memory usage information
    """
    if not GPU_AVAILABLE:
        return {'available': False}
    
    result = {'available': True}
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        result['torch'] = {
            'allocated': torch.cuda.memory_allocated(current_device) / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved(current_device) / 1024**2,  # MB
            'max_allocated': torch.cuda.max_memory_allocated(current_device) / 1024**2  # MB
        }
    
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        result['cupy'] = {
            'used': mempool.used_bytes() / 1024**2,  # MB
            'total': mempool.total_bytes() / 1024**2  # MB
        }
    
    return result

def clear_gpu_memory():
    """Clear GPU memory cache to free up resources"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
    print("GPU memory cache cleared.")

if __name__ == "__main__":
    # Run self-test if executed directly
    print("\nGPU Acceleration Summary:")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Use GPU Enabled: {USE_GPU}")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"CuPy Available: {CUPY_AVAILABLE}")
    print(f"cuSpatial Available: {CUSPATIAL_AVAILABLE}")
    
    if GPU_AVAILABLE:
        print("\nCurrent GPU Memory Usage:")
        print(memory_usage())
        
        print("\nSimple Performance Test:")
        
        # Create large arrays
        size = 10000000
        
        # Define test function
        def test_func(size):
            a = get_array_module().__array(np.random.random(size))
            b = get_array_module().__array(np.random.random(size))
            c = a + b
            d = a * b
            e = c / d
            return e.mean()
        
        results = time_function(test_func, size, repeat=5)
        print(f"CPU time: {results['cpu']['mean']:.6f} seconds")
        if 'gpu' in results:
            print(f"GPU time: {results['gpu']['mean']:.6f} seconds")
            print(f"Speedup: {results['speedup']:.2f}x")
        
        clear_gpu_memory() 