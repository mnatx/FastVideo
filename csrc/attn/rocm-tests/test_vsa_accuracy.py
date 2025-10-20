#!/usr/bin/env python3
"""
Final Comprehensive VSA Accuracy Test for ROCm Platform

This test provides a complete accuracy evaluation of the Video Sparse Attention backend
on ROCm, sweeping through various configurations and comparing against PyTorch reference.

GPU-Specific Optimizations:
- AMD Radeon PRO W7800: Conservative configurations optimized for 30GB memory
- AMD Instinct MI250: Aggressive configurations optimized for 128GB memory
- Generic ROCm: Balanced configurations for other ROCm-compatible GPUs

The test automatically detects the GPU type and selects appropriate parameter ranges.
"""

import torch
import sys
import os
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns

# Import VSA package
try:
    import vsa
    VSA_AVAILABLE = True
    print("✅ VSA package imported successfully")
except ImportError as e:
    print(f"❌ VSA package not available: {e}")
    VSA_AVAILABLE = False
    sys.exit(1)

@dataclass
class TestConfig:
    """Configuration for a single test case."""
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    top_k: int
    sparsity: float = 0.5
    dtype: torch.dtype = torch.bfloat16

@dataclass
class AccuracyMetrics:
    """Accuracy metrics for comparing VSA vs PyTorch reference."""
    output_mae: float
    output_mse: float
    output_max_error: float
    output_relative_error: float
    
    grad_q_mae: float
    grad_q_mse: float
    grad_q_max_error: float
    grad_q_relative_error: float
    
    grad_k_mae: float
    grad_k_mse: float
    grad_k_max_error: float
    grad_k_relative_error: float
    
    grad_v_mae: float
    grad_v_mse: float
    grad_v_max_error: float
    grad_v_relative_error: float

class ComprehensiveVSATester:
    """Comprehensive VSA accuracy tester for ROCm."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.is_rocm = self._is_rocm_platform()
        self.results = []
        
    def _is_rocm_platform(self) -> bool:
        """Check if we're running on ROCm platform."""
        return (torch.cuda.is_available() and 
                hasattr(torch.version, 'hip') and 
                torch.version.hip is not None)
    
    def _get_platform_info(self) -> str:
        """Get platform information."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            if self.is_rocm:
                hip_version = torch.version.hip
                gpu_type = self._detect_gpu_type()
                return f"ROCm (HIP {hip_version}) - {device_name} ({device_count} devices) [Type: {gpu_type.upper()}]"
            else:
                return f"CUDA - {device_name} ({device_count} devices)"
        else:
            return "CPU only"
    
    def _generate_block_sparse_mask(self, h: int, num_blocks: int, k: int, device: str = "cuda"):
        """Generate block sparse mask."""
        k = min(k, num_blocks)
        scores = torch.rand(h, num_blocks, num_blocks, device=device)
        _, indices = torch.topk(scores, k, dim=-1)
        block_sparse_mask = torch.zeros(h, num_blocks, num_blocks, dtype=torch.bool, device=device)
        block_sparse_mask = block_sparse_mask.scatter_(2, indices, 1).bool()
        return block_sparse_mask
    
    def _create_full_mask_from_block_mask(self, block_sparse_mask, variable_block_sizes, device="cuda"):
        """Convert block-level sparse mask to full attention mask."""
        h, num_blocks, _ = block_sparse_mask.shape
        total_seq_len = variable_block_sizes.sum().item()
        cumsum = torch.cat([torch.tensor([0], device=device), variable_block_sizes.cumsum(dim=0)[:-1]])

        full_mask = torch.zeros(h, total_seq_len, total_seq_len, dtype=torch.bool, device=device)

        for head in range(h):
            for q_block in range(num_blocks):
                q_start = cumsum[q_block]
                q_end = q_start + variable_block_sizes[q_block]

                for kv_block in range(num_blocks):
                    if block_sparse_mask[head, q_block, kv_block]:
                        kv_start = cumsum[kv_block]
                        kv_end = kv_start + variable_block_sizes[kv_block]
                        full_mask[head, q_start:q_end, kv_start:kv_end] = True

        return full_mask
    
    def _get_non_pad_index(self, vid_len: torch.LongTensor, n_win: int, win_size: int):
        """Get non-padding indices."""
        device = vid_len.device
        starts_pad = torch.arange(n_win, device=device) * win_size
        index_pad = starts_pad[:, None] + torch.arange(win_size, device=device)[None, :]
        index_mask = torch.arange(win_size, device=device)[None, :] < vid_len[:, None]
        return index_pad[index_mask]
    
    def _generate_variable_block_sizes(self, num_blocks, min_size=32, max_size=64, device="cuda"):
        """Generate variable block sizes."""
        return torch.randint(min_size, max_size + 1, (num_blocks,), device=device, dtype=torch.int32)
    
    def _vsa_pad(self, x, non_pad_index, num_blocks, block_size):
        """Pad tensor for VSA."""
        padded_x = torch.zeros((1, x.shape[1], num_blocks * block_size, x.shape[3]), 
                              device=x.device, dtype=x.dtype)
        padded_x[:, :, non_pad_index, :] = x
        return padded_x
    
    def _pytorch_reference(self, Q, K, V, block_sparse_mask, dO):
        """PyTorch reference implementation."""
        q_ = Q.clone().float().requires_grad_()
        k_ = K.clone().float().requires_grad_()
        v_ = V.clone().float().requires_grad_()

        QK = torch.matmul(q_, k_.transpose(-2, -1))
        QK /= (q_.size(-1) ** 0.5)
        QK = QK.masked_fill(~block_sparse_mask.unsqueeze(0), float('-inf'))

        QK = torch.nn.functional.softmax(QK, dim=-1)
        output = torch.matmul(QK, v_)

        dO_ = dO
        output.backward(dO_)
        return (
            output.to(torch.bfloat16),
            q_.grad.to(torch.bfloat16),
            k_.grad.to(torch.bfloat16),
            v_.grad.to(torch.bfloat16),
        )
    
    def _vsa_implementation(self, Q, K, V, block_sparse_mask, variable_block_sizes, non_pad_index, dO):
        """VSA implementation."""
        Q = Q.detach().requires_grad_()
        K = K.detach().requires_grad_()
        V = V.detach().requires_grad_()
        
        block_size = 64  # VSA requires 64
        q_padded = self._vsa_pad(Q, non_pad_index, variable_block_sizes.shape[0], block_size)
        k_padded = self._vsa_pad(K, non_pad_index, variable_block_sizes.shape[0], block_size)
        v_padded = self._vsa_pad(V, non_pad_index, variable_block_sizes.shape[0], block_size)
        
        output, _ = vsa.block_sparse_attn(q_padded, k_padded, v_padded, block_sparse_mask, variable_block_sizes)
        output = output[:, :, non_pad_index, :]
        output.backward(dO)
        return output, Q.grad, K.grad, V.grad
    
    def _compute_accuracy_metrics(self, ref_tensor: torch.Tensor, test_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute accuracy metrics between reference and test tensors."""
        if ref_tensor is None or test_tensor is None:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'max_error': float('inf'),
                'relative_error': float('inf')
            }
        
        # Convert to float for accurate comparison
        ref_float = ref_tensor.float()
        test_float = test_tensor.float()
        
        # Compute differences
        diff = ref_float - test_float
        abs_diff = torch.abs(diff)
        
        # Mean Absolute Error
        mae = torch.mean(abs_diff).item()
        
        # Mean Squared Error
        mse = torch.mean(diff ** 2).item()
        
        # Maximum error
        max_error = torch.max(abs_diff).item()
        
        # Relative error (normalized by reference magnitude)
        ref_magnitude = torch.mean(torch.abs(ref_float))
        relative_error = mae / (ref_magnitude.item() + 1e-8)
        
        return {
            'mae': mae,
            'mse': mse,
            'max_error': max_error,
            'relative_error': relative_error
        }
    
    def _generate_test_data(self, config: TestConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test data for the given configuration."""
        # Generate random tensors
        q = torch.randn(config.batch_size, config.num_heads, config.seq_len, config.head_dim,
                       device=self.device, dtype=config.dtype)
        k = torch.randn(config.batch_size, config.num_heads, config.seq_len, config.head_dim,
                       device=self.device, dtype=config.dtype)
        v = torch.randn(config.batch_size, config.num_heads, config.seq_len, config.head_dim,
                       device=self.device, dtype=config.dtype)
        dO = torch.randn(config.batch_size, config.num_heads, config.seq_len, config.head_dim,
                        device=self.device, dtype=config.dtype)
        
        return q, k, v, dO
    
    def run_single_test(self, config: TestConfig) -> Optional[AccuracyMetrics]:
        """Run a single test case and return accuracy metrics."""
        try:
            # Generate test data
            Q, K, V, dO = self._generate_test_data(config)
            
            # Calculate number of blocks based on sequence length and block size
            block_size = 64  # VSA requires 64
            num_blocks = max(1, config.seq_len // block_size)
            top_k = min(config.top_k, num_blocks)
            
            # Generate variable block sizes
            variable_block_sizes = self._generate_variable_block_sizes(num_blocks, device=self.device)
            S = int(variable_block_sizes.sum().item())
            padded_S = num_blocks * block_size
            non_pad_index = self._get_non_pad_index(variable_block_sizes, num_blocks, block_size)
            
            # Generate block sparse mask
            block_mask = self._generate_block_sparse_mask(config.num_heads, num_blocks, top_k, device=self.device)
            full_mask = self._create_full_mask_from_block_mask(block_mask, variable_block_sizes, device=self.device)
            
            # Adjust tensors to match actual sequence length
            Q = Q[:, :, :S, :]
            K = K[:, :, :S, :]
            V = V[:, :, :S, :]
            dO = dO[:, :, :S, :]
            
            # Run PyTorch reference
            pt_o, pt_qg, pt_kg, pt_vg = self._pytorch_reference(Q, K, V, full_mask, dO)
            
            # Run VSA implementation
            vsa_o, vsa_qg, vsa_kg, vsa_vg = self._vsa_implementation(Q, K, V, block_mask.unsqueeze(0), 
                                                                   variable_block_sizes, non_pad_index, dO)
            
            # Compute accuracy metrics
            output_metrics = self._compute_accuracy_metrics(pt_o, vsa_o)
            grad_q_metrics = self._compute_accuracy_metrics(pt_qg, vsa_qg)
            grad_k_metrics = self._compute_accuracy_metrics(pt_kg, vsa_kg)
            grad_v_metrics = self._compute_accuracy_metrics(pt_vg, vsa_vg)
            
            return AccuracyMetrics(
                output_mae=output_metrics['mae'],
                output_mse=output_metrics['mse'],
                output_max_error=output_metrics['max_error'],
                output_relative_error=output_metrics['relative_error'],
                
                grad_q_mae=grad_q_metrics['mae'],
                grad_q_mse=grad_q_metrics['mse'],
                grad_q_max_error=grad_q_metrics['max_error'],
                grad_q_relative_error=grad_q_metrics['relative_error'],
                
                grad_k_mae=grad_k_metrics['mae'],
                grad_k_mse=grad_k_metrics['mse'],
                grad_k_max_error=grad_k_metrics['max_error'],
                grad_k_relative_error=grad_k_metrics['relative_error'],
                
                grad_v_mae=grad_v_metrics['mae'],
                grad_v_mse=grad_v_metrics['mse'],
                grad_v_max_error=grad_v_metrics['max_error'],
                grad_v_relative_error=grad_v_metrics['relative_error']
            )
            
        except Exception as e:
            print(f"Error in test case {config}: {e}")
            return None
    
    def _detect_gpu_type(self) -> str:
        """Detect the type of GPU being used."""
        if not torch.cuda.is_available():
            return "unknown"
        
        device_name = torch.cuda.get_device_name().lower()
        if "mi250" in device_name or "instinct" in device_name:
            return "mi250"
        elif "w7800" in device_name or "radeon pro" in device_name:
            return "w7800"
        else:
            return "generic_rocm"
    
    def _validate_config_for_gpu(self, config: TestConfig) -> bool:
        """Validate if a configuration is appropriate for the detected GPU."""
        if not self.is_rocm:
            return True
        
        gpu_type = self._detect_gpu_type()
        total_elements = config.batch_size * config.num_heads * config.seq_len * config.head_dim
        
        if gpu_type == "mi250":
            # MI250 can handle very large configurations
            return total_elements <= 10000000  # 10M elements
        elif gpu_type == "w7800":
            # W7800 has more conservative limits
            return total_elements <= 1000000   # 1M elements
        else:
            # Generic ROCm - balanced approach
            return total_elements <= 2000000   # 2M elements
    
    def generate_comprehensive_test_configs(self) -> List[TestConfig]:
        """Generate comprehensive test configurations."""
        configs = []
        
        if self.is_rocm:
            gpu_type = self._detect_gpu_type()
            
            if gpu_type == "mi250":
                # MI250 configurations - optimized for larger memory capacity
                # MI250 has 128GB memory vs W7800's 30GB, allowing for more aggressive configs
                batch_sizes = [1, 2, 4, 8]  # Increased batch sizes
                num_heads_list = [4, 8, 16, 24, 32]  # More heads
                seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]  # Longer sequences
                head_dims = [64, 128, 256]  # More head dimensions
                top_k_list = [1, 2, 4, 8, 16, 32, 64]  # More top-k values
            elif gpu_type == "w7800":
                # W7800 configurations - conservative for shared memory
                batch_sizes = [1, 2]
                num_heads_list = [2, 4, 8]
                seq_lens = [64, 128, 256, 384, 512]  # Must be multiples of 64
                head_dims = [64, 128]  # VSA supports both
                top_k_list = [1, 2, 4, 8, 16]
            else:
                # Generic ROCm configurations - balanced approach
                batch_sizes = [1, 2, 4]
                num_heads_list = [4, 8, 16]
                seq_lens = [64, 128, 256, 512, 1024]
                head_dims = [64, 128]
                top_k_list = [2, 4, 8, 16, 32]
        else:
            # CUDA configurations
            batch_sizes = [1, 2, 4]
            num_heads_list = [4, 8, 16, 32]
            seq_lens = [64, 128, 256, 512, 1024]
            head_dims = [64, 128, 256]
            top_k_list = [2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            for num_heads in num_heads_list:
                for seq_len in seq_lens:
                    for head_dim in head_dims:
                        for top_k in top_k_list:
                            # Skip invalid configurations
                            block_size = 64
                            if top_k > seq_len // block_size:
                                continue
                            
                            # Skip configurations that might cause memory issues
                            test_config = TestConfig(
                                batch_size=batch_size,
                                num_heads=num_heads,
                                seq_len=seq_len,
                                head_dim=head_dim,
                                top_k=top_k
                            )
                            
                            if not self._validate_config_for_gpu(test_config):
                                continue
                            
                            configs.append(test_config)
        
        return configs
    
    def run_comprehensive_test(self, max_configs: Optional[int] = None) -> Dict:
        """Run comprehensive accuracy test."""
        print("=" * 80)
        print("Comprehensive VSA Accuracy Test for ROCm")
        print("=" * 80)
        print(f"Platform: {self._get_platform_info()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm optimized: {self.is_rocm}")
        if self.is_rocm:
            gpu_type = self._detect_gpu_type()
            print(f"GPU Type detected: {gpu_type.upper()}")
            if gpu_type == "mi250":
                print("Using MI250-optimized configurations (larger batch sizes, longer sequences)")
            elif gpu_type == "w7800":
                print("Using W7800-optimized configurations (conservative memory usage)")
            else:
                print("Using generic ROCm configurations (balanced approach)")
        print()
        
        if not VSA_AVAILABLE:
            print("❌ VSA package not available")
            return {}
        
        # Generate test configurations
        configs = self.generate_comprehensive_test_configs()
        if max_configs:
            configs = configs[:max_configs]
        
        print(f"Running {len(configs)} test configurations...")
        
        # Show configuration ranges
        if configs:
            batch_sizes = sorted(set(c.batch_size for c in configs))
            num_heads = sorted(set(c.num_heads for c in configs))
            seq_lens = sorted(set(c.seq_len for c in configs))
            head_dims = sorted(set(c.head_dim for c in configs))
            top_k_vals = sorted(set(c.top_k for c in configs))
            
            print(f"Configuration ranges:")
            print(f"  Batch sizes: {batch_sizes}")
            print(f"  Number of heads: {num_heads}")
            print(f"  Sequence lengths: {seq_lens}")
            print(f"  Head dimensions: {head_dims}")
            print(f"  Top-k values: {top_k_vals}")
        print()
        
        # Run tests
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, config in enumerate(tqdm(configs, desc="Running tests")):
            metrics = self.run_single_test(config)
            
            if metrics is not None:
                results.append({
                    'config': {
                        'batch_size': config.batch_size,
                        'num_heads': config.num_heads,
                        'seq_len': config.seq_len,
                        'head_dim': config.head_dim,
                        'top_k': config.top_k,
                        'sparsity': config.sparsity
                    },
                    'metrics': {
                        'output_mae': metrics.output_mae,
                        'output_mse': metrics.output_mse,
                        'output_max_error': metrics.output_max_error,
                        'output_relative_error': metrics.output_relative_error,
                        'grad_q_mae': metrics.grad_q_mae,
                        'grad_q_mse': metrics.grad_q_mse,
                        'grad_q_max_error': metrics.grad_q_max_error,
                        'grad_q_relative_error': metrics.grad_q_relative_error,
                        'grad_k_mae': metrics.grad_k_mae,
                        'grad_k_mse': metrics.grad_k_mse,
                        'grad_k_max_error': metrics.grad_k_max_error,
                        'grad_k_relative_error': metrics.grad_k_relative_error,
                        'grad_v_mae': metrics.grad_v_mae,
                        'grad_v_mse': metrics.grad_v_mse,
                        'grad_v_max_error': metrics.grad_v_max_error,
                        'grad_v_relative_error': metrics.grad_v_relative_error
                    }
                })
                successful_tests += 1
                if (i + 1) % 10 == 0:
                    print(f"✅ Completed {i+1}/{len(configs)} tests")
            else:
                failed_tests += 1
                if (i + 1) % 10 == 0:
                    print(f"❌ Failed {i+1}/{len(configs)} tests")
            
            # Clear cache periodically
            if (i + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Generate summary
        summary = self._generate_summary(results, successful_tests, failed_tests)
        
        return {
            'summary': summary,
            'results': results,
            'platform_info': self._get_platform_info(),
            'pytorch_version': torch.__version__,
            'rocm_optimized': self.is_rocm
        }
    
    def _generate_summary(self, results: List[Dict], successful_tests: int, failed_tests: int) -> Dict:
        """Generate test summary statistics."""
        if not results:
            return {
                'total_tests': successful_tests + failed_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': 0.0,
                'error': 'No successful tests'
            }
        
        # Collect all metrics
        all_output_mae = [r['metrics']['output_mae'] for r in results]
        all_output_relative_error = [r['metrics']['output_relative_error'] for r in results]
        all_grad_q_mae = [r['metrics']['grad_q_mae'] for r in results]
        all_grad_k_mae = [r['metrics']['grad_k_mae'] for r in results]
        all_grad_v_mae = [r['metrics']['grad_v_mae'] for r in results]
        
        return {
            'total_tests': successful_tests + failed_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / (successful_tests + failed_tests) if (successful_tests + failed_tests) > 0 else 0,
            'output_mae': {
                'mean': np.mean(all_output_mae),
                'std': np.std(all_output_mae),
                'min': np.min(all_output_mae),
                'max': np.max(all_output_mae),
                'median': np.median(all_output_mae)
            },
            'output_relative_error': {
                'mean': np.mean(all_output_relative_error),
                'std': np.std(all_output_relative_error),
                'min': np.min(all_output_relative_error),
                'max': np.max(all_output_relative_error),
                'median': np.median(all_output_relative_error)
            },
            'grad_q_mae': {
                'mean': np.mean(all_grad_q_mae),
                'std': np.std(all_grad_q_mae),
                'min': np.min(all_grad_q_mae),
                'max': np.max(all_grad_q_mae),
                'median': np.median(all_grad_q_mae)
            },
            'grad_k_mae': {
                'mean': np.mean(all_grad_k_mae),
                'std': np.std(all_grad_k_mae),
                'min': np.min(all_grad_k_mae),
                'max': np.max(all_grad_k_mae),
                'median': np.median(all_grad_k_mae)
            },
            'grad_v_mae': {
                'mean': np.mean(all_grad_v_mae),
                'std': np.std(all_grad_v_mae),
                'min': np.min(all_grad_v_mae),
                'max': np.max(all_grad_v_mae),
                'median': np.median(all_grad_v_mae)
            }
        }
    
    def print_comprehensive_summary(self, summary: Dict, results: List[Dict]):
        """Print comprehensive test summary."""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE VSA ACCURACY TEST SUMMARY")
        print("=" * 100)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print()
        
        print("OUTPUT ACCURACY METRICS:")
        print(f"  Mean Absolute Error: {summary['output_mae']['mean']:.6e} ± {summary['output_mae']['std']:.6e}")
        print(f"  Relative Error: {summary['output_relative_error']['mean']:.6e} ± {summary['output_relative_error']['std']:.6e}")
        print(f"  MAE Range: [{summary['output_mae']['min']:.6e}, {summary['output_mae']['max']:.6e}]")
        print()
        
        print("GRADIENT ACCURACY METRICS:")
        print(f"  Query Gradients MAE: {summary['grad_q_mae']['mean']:.6e} ± {summary['grad_q_mae']['std']:.6e}")
        print(f"  Key Gradients MAE: {summary['grad_k_mae']['mean']:.6e} ± {summary['grad_k_mae']['std']:.6e}")
        print(f"  Value Gradients MAE: {summary['grad_v_mae']['mean']:.6e} ± {summary['grad_v_mae']['std']:.6e}")
        print()
        
        # Analyze by configuration parameters
        self._analyze_by_parameters(results)
        
        # Print detailed results table (first 20)
        print("\nDETAILED RESULTS TABLE (First 20 tests):")
        print("-" * 120)
        print(f"{'Test':<4} {'Batch':<5} {'Heads':<5} {'SeqLen':<6} {'HeadDim':<7} {'TopK':<4} {'Output MAE':<12} {'Grad Q MAE':<12} {'Grad K MAE':<12} {'Grad V MAE':<12}")
        print("-" * 120)
        
        for i, result in enumerate(results[:20]):
            config = result['config']
            metrics = result['metrics']
            print(f"{i+1:<4} {config['batch_size']:<5} {config['num_heads']:<5} {config['seq_len']:<6} {config['head_dim']:<7} {config['top_k']:<4} "
                  f"{metrics['output_mae']:<12.6e} {metrics['grad_q_mae']:<12.6e} {metrics['grad_k_mae']:<12.6e} {metrics['grad_v_mae']:<12.6e}")
        
        if len(results) > 20:
            print(f"... and {len(results) - 20} more tests")
    
    def _analyze_by_parameters(self, results: List[Dict]):
        """Analyze accuracy by different configuration parameters."""
        print("ANALYSIS BY CONFIGURATION PARAMETERS:")
        print("-" * 50)
        
        # Group by sequence length
        seq_len_groups = {}
        for result in results:
            seq_len = result['config']['seq_len']
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append(result['metrics']['output_mae'])
        
        print("Output MAE by Sequence Length:")
        for seq_len in sorted(seq_len_groups.keys()):
            mae_values = seq_len_groups[seq_len]
            print(f"  {seq_len:3d}: {np.mean(mae_values):.6e} ± {np.std(mae_values):.6e} (n={len(mae_values)})")
        
        # Group by number of heads
        heads_groups = {}
        for result in results:
            num_heads = result['config']['num_heads']
            if num_heads not in heads_groups:
                heads_groups[num_heads] = []
            heads_groups[num_heads].append(result['metrics']['output_mae'])
        
        print("\nOutput MAE by Number of Heads:")
        for num_heads in sorted(heads_groups.keys()):
            mae_values = heads_groups[num_heads]
            print(f"  {num_heads:2d}: {np.mean(mae_values):.6e} ± {np.std(mae_values):.6e} (n={len(mae_values)})")
        
        # Group by top-k
        topk_groups = {}
        for result in results:
            top_k = result['config']['top_k']
            if top_k not in topk_groups:
                topk_groups[top_k] = []
            topk_groups[top_k].append(result['metrics']['output_mae'])
        
        print("\nOutput MAE by Top-K:")
        for top_k in sorted(topk_groups.keys()):
            mae_values = topk_groups[top_k]
            print(f"  {top_k:2d}: {np.mean(mae_values):.6e} ± {np.std(mae_values):.6e} (n={len(mae_values)})")

def main():
    """Main function to run the comprehensive test."""
    parser = argparse.ArgumentParser(description='Comprehensive VSA Accuracy Test for ROCm')
    parser.add_argument('--max-configs', type=int, default=50, 
                       help='Maximum number of configurations to test')
    parser.add_argument('--output-file', type=str, default='vsa_accuracy_results_comprehensive.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run tests on')
    
    args = parser.parse_args()
    
    # Create tester
    tester = ComprehensiveVSATester(device=args.device)
    
    # Run comprehensive test
    start_time = time.time()
    results = tester.run_comprehensive_test(max_configs=args.max_configs)
    end_time = time.time()
    
    if results and results['results']:
        # Print comprehensive summary
        tester.print_comprehensive_summary(results['summary'], results['results'])
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_file}")
        
        print(f"\nTotal test time: {end_time - start_time:.2f} seconds")
        
        # Print final assessment
        print("\n" + "=" * 100)
        print("FINAL ASSESSMENT")
        print("=" * 100)
        success_rate = results['summary']['success_rate']
        output_mae = results['summary']['output_mae']['mean']
        
        # Show GPU-specific information
        if self.is_rocm:
            gpu_type = self._detect_gpu_type()
            print(f"GPU Type: {gpu_type.upper()}")
            if gpu_type == "mi250":
                print("✅ MI250-optimized configurations used (128GB memory, aggressive parameters)")
            elif gpu_type == "w7800":
                print("✅ W7800-optimized configurations used (30GB memory, conservative parameters)")
            else:
                print("✅ Generic ROCm configurations used (balanced parameters)")
            print()
        
        if success_rate >= 0.9 and output_mae < 1e-3:
            print("✅ EXCELLENT: VSA implementation shows high accuracy and reliability")
        elif success_rate >= 0.8 and output_mae < 1e-2:
            print("✅ GOOD: VSA implementation shows good accuracy and reliability")
        elif success_rate >= 0.7:
            print("⚠️  ACCEPTABLE: VSA implementation shows acceptable accuracy with some issues")
        else:
            print("❌ NEEDS IMPROVEMENT: VSA implementation has significant accuracy issues")
        
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Output MAE: {output_mae:.6e}")
        
    else:
        print("❌ Test failed - no results generated")

if __name__ == "__main__":
    main()