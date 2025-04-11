"""
Unit tests for the GPU utilities module
"""

import os
import pytest
import numpy as np
import sys

# Ensure src directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from src.gpu_utils import (
    get_array_module, 
    to_device, 
    to_cpu, 
    time_function, 
    GPU_AVAILABLE,
    TORCH_AVAILABLE,
    CUPY_AVAILABLE
)

# Skip tests that require GPU if no GPU is available
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
requires_cupy = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")

class TestGPUUtils:
    """Test suite for GPU utilities module"""
    
    def test_get_array_module_cpu(self):
        """Test that get_array_module returns numpy when use_gpu=False"""
        xp = get_array_module(use_gpu=False)
        assert xp == np
    
    @requires_cupy
    def test_get_array_module_gpu(self):
        """Test that get_array_module returns cupy when use_gpu=True"""
        import cupy as cp
        xp = get_array_module(use_gpu=True)
        assert xp == cp
    
    def test_to_device_numpy(self):
        """Test to_device with numpy array"""
        arr = np.array([1, 2, 3])
        result = to_device(arr, use_gpu=False)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    @requires_torch
    def test_to_device_torch(self):
        """Test to_device with torch tensor"""
        import torch
        tensor = torch.tensor([1, 2, 3])
        result = to_device(tensor, use_gpu=False)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cpu'
    
    @requires_gpu
    @requires_torch
    def test_to_device_torch_gpu(self):
        """Test to_device with torch tensor and GPU"""
        import torch
        tensor = torch.tensor([1, 2, 3])
        result = to_device(tensor, use_gpu=True)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cuda'
    
    @requires_torch
    def test_to_cpu_torch(self):
        """Test to_cpu with torch tensor"""
        import torch
        tensor = torch.tensor([1, 2, 3])
        result = to_cpu(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    @requires_cupy
    def test_to_cpu_cupy(self):
        """Test to_cpu with cupy array"""
        import cupy as cp
        arr = cp.array([1, 2, 3])
        result = to_cpu(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    def test_time_function(self):
        """Test timing a function"""
        def dummy_func(n):
            """A dummy function that just wastes time"""
            sum = 0
            for i in range(n):
                sum += i
            return sum
        
        results = time_function(dummy_func, 1000000, repeat=2)
        
        # Check that we have CPU timings
        assert 'cpu' in results
        assert 'mean' in results['cpu']
        assert 'min' in results['cpu']
        assert 'max' in results['cpu']
        
        # Check that all times are positive
        assert results['cpu']['mean'] > 0
        assert results['cpu']['min'] > 0
        assert results['cpu']['max'] > 0
        
        # If GPU is available, check GPU timings too
        if GPU_AVAILABLE:
            assert 'gpu' in results
            assert 'mean' in results['gpu']
            assert 'speedup' in results
    
    @pytest.mark.parametrize("test_input,expected", [
        # Test various inputs
        (np.array([1, 2, 3]), np.array([1, 2, 3])),  # numpy array
        ([1, 2, 3], [1, 2, 3]),  # list
        (1, 1),  # scalar
    ])
    def test_to_cpu_various_inputs(self, test_input, expected):
        """Test to_cpu with various input types"""
        result = to_cpu(test_input)
        
        # For numpy arrays, use numpy's testing utilities
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(result, expected)
        else:
            assert result == expected


if __name__ == "__main__":
    # Run the tests manually if executed directly
    pytest.main(["-v", __file__]) 