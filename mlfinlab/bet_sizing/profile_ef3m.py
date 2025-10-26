"""
Profiling script for EF3M to measure current performance and identify bottlenecks.
This will help determine if Rust migration is necessary.
"""

import time
import sys
import os
import numpy as np
import pandas as pd
import cProfile
import pstats
from io import StringIO

# Add the mlfinlab directory to the path
sys.path.insert(0, '/app/scripts/jmrichardson_mlfinlab')

# Import directly from the module
from mlfinlab.bet_sizing.ef3m import M2N, raw_moment

def generate_test_data():
    """Generate realistic test data for EF3M."""
    # Simulate returns from a mixture of two Gaussians
    np.random.seed(42)
    
    # Parameters for mixture
    mu1, mu2 = 0.02, -0.01
    sigma1, sigma2 = 0.05, 0.08
    p1 = 0.6
    
    # Generate samples
    n_samples = 10000
    mask = np.random.rand(n_samples) < p1
    samples = np.where(
        mask,
        np.random.normal(mu1, sigma1, n_samples),
        np.random.normal(mu2, sigma2, n_samples)
    )
    
    # Calculate first 5 raw moments
    moments = [np.mean(samples**i) for i in range(1, 6)]
    
    return moments, (mu1, mu2, sigma1, sigma2, p1)

def benchmark_single_run(moments, epsilon=1e-5, variant=1):
    """Benchmark a single EF3M run."""
    print(f"\n{'='*60}")
    print(f"Benchmarking EF3M Variant {variant}")
    print(f"{'='*60}")
    
    m2n = M2N(moments, epsilon=epsilon, factor=5, n_runs=1, variant=variant, max_iter=100_000)
    
    start = time.perf_counter()
    result = m2n.mp_fit()
    elapsed = time.perf_counter() - start
    
    print(f"\n⏱️  Single run (1 iteration): {elapsed*1000:.2f}ms")
    print(f"📊 Result shape: {result.shape}")
    print(f"✅ Parameters found: {len(result) > 0}")
    
    if len(result) > 0:
        print(f"\nEstimated parameters:")
        print(result.to_string())
    
    return elapsed, result

def benchmark_parallel_runs(moments, n_runs_list=[1, 5, 10, 20], epsilon=1e-5, variant=1):
    """Benchmark parallel execution with different n_runs."""
    print(f"\n{'='*60}")
    print(f"Parallel Execution Benchmark (Variant {variant})")
    print(f"{'='*60}")
    
    results = []
    
    for n_runs in n_runs_list:
        m2n = M2N(moments, epsilon=epsilon, factor=5, n_runs=n_runs, variant=variant, max_iter=100_000)
        
        start = time.perf_counter()
        result = m2n.mp_fit()
        elapsed = time.perf_counter() - start
        
        per_run = elapsed / n_runs if n_runs > 0 else 0
        
        print(f"\nn_runs={n_runs:3d}: {elapsed*1000:7.2f}ms total | {per_run*1000:6.2f}ms per run")
        
        results.append({
            'n_runs': n_runs,
            'total_time_ms': elapsed * 1000,
            'per_run_ms': per_run * 1000,
            'result_rows': len(result)
        })
    
    return pd.DataFrame(results)

def profile_detailed(moments, epsilon=1e-5, n_runs=5, variant=1):
    """Run detailed profiling to identify bottlenecks."""
    print(f"\n{'='*60}")
    print(f"Detailed Profiling (Variant {variant}, n_runs={n_runs})")
    print(f"{'='*60}")
    
    profiler = cProfile.Profile()
    
    m2n = M2N(moments, epsilon=epsilon, factor=5, n_runs=n_runs, variant=variant, max_iter=100_000)
    
    # Profile the execution
    profiler.enable()
    result = m2n.mp_fit()
    profiler.disable()
    
    # Print statistics
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\n🔍 Top 20 functions by cumulative time:")
    print(s.getvalue())
    
    return result

def test_accuracy(moments, true_params, epsilon=1e-5, variant=1):
    """Test how accurately EF3M recovers the true parameters."""
    print(f"\n{'='*60}")
    print(f"Accuracy Test (Variant {variant})")
    print(f"{'='*60}")
    
    m2n = M2N(moments, epsilon=epsilon, factor=5, n_runs=20, variant=variant, max_iter=100_000)
    result = m2n.mp_fit()
    
    if len(result) > 0:
        # Get the best result (lowest error)
        best = result.loc[result['error'].idxmin()]
        
        mu1_true, mu2_true, sigma1_true, sigma2_true, p1_true = true_params
        
        print(f"\n📈 True parameters:")
        print(f"  mu1={mu1_true:.4f}, mu2={mu2_true:.4f}")
        print(f"  sigma1={sigma1_true:.4f}, sigma2={sigma2_true:.4f}")
        print(f"  p1={p1_true:.4f}")
        
        print(f"\n📉 Estimated parameters:")
        print(f"  mu1={best['mu_1']:.4f}, mu2={best['mu_2']:.4f}")
        print(f"  sigma1={best['sigma_1']:.4f}, sigma2={best['sigma_2']:.4f}")
        print(f"  p1={best['p_1']:.4f}")
        
        print(f"\n❌ Errors:")
        print(f"  mu1: {abs(best['mu_1'] - mu1_true):.6f}")
        print(f"  mu2: {abs(best['mu_2'] - mu2_true):.6f}")
        print(f"  sigma1: {abs(best['sigma_1'] - sigma1_true):.6f}")
        print(f"  sigma2: {abs(best['sigma_2'] - sigma2_true):.6f}")
        print(f"  p1: {abs(best['p_1'] - p1_true):.6f}")
        print(f"  fitting_error: {best['error']:.6e}")

def compare_variants(moments, epsilon=1e-5, n_runs=10):
    """Compare performance of variant 1 vs variant 2."""
    print(f"\n{'='*60}")
    print(f"Variant Comparison")
    print(f"{'='*60}")
    
    for variant in [1, 2]:
        m2n = M2N(moments, epsilon=epsilon, factor=5, n_runs=n_runs, variant=variant, max_iter=100_000)
        
        start = time.perf_counter()
        result = m2n.mp_fit()
        elapsed = time.perf_counter() - start
        
        best_error = result['error'].min() if len(result) > 0 else float('inf')
        
        print(f"\nVariant {variant}:")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Best error: {best_error:.6e}")
        print(f"  Results found: {len(result)}")

def main():
    """Run all benchmarks and profiling."""
    print("\n" + "="*60)
    print("EF3M Performance Analysis")
    print("="*60)
    
    # Generate test data
    print("\n📊 Generating test data...")
    moments, true_params = generate_test_data()
    print(f"✅ Generated moments: {[f'{m:.6f}' for m in moments]}")
    
    # 1. Single run benchmark
    elapsed_single, _ = benchmark_single_run(moments, epsilon=1e-5, variant=1)
    
    # 2. Parallel execution benchmark
    parallel_results = benchmark_parallel_runs(moments, n_runs_list=[1, 5, 10, 20, 50], epsilon=1e-5, variant=1)
    print(f"\n📊 Parallel scaling summary:")
    print(parallel_results.to_string(index=False))
    
    # 3. Detailed profiling
    profile_detailed(moments, epsilon=1e-5, n_runs=5, variant=1)
    
    # 4. Accuracy test
    test_accuracy(moments, true_params, epsilon=1e-5, variant=1)
    
    # 5. Compare variants
    compare_variants(moments, epsilon=1e-5, n_runs=10)
    
    # Summary and recommendations
    print(f"\n{'='*60}")
    print("📝 Summary & Rust Migration Assessment")
    print(f"{'='*60}")
    print(f"\n⏱️  Current Performance:")
    print(f"  Single iteration: ~{elapsed_single*1000:.1f}ms")
    print(f"  Typical use (n_runs=20): ~{elapsed_single*20*1000:.1f}ms")
    
    print(f"\n🎯 Latency Requirements:")
    print(f"  Real-time trading (< 10ms): {'❌ RUST NEEDED' if elapsed_single*20 > 0.01 else '✅ OK'}")
    print(f"  Low latency (< 100ms): {'❌ RUST NEEDED' if elapsed_single*20 > 0.1 else '✅ OK'}")
    print(f"  Batch processing (< 1s): {'❌ RUST NEEDED' if elapsed_single*20 > 1.0 else '✅ OK'}")
    
    print(f"\n💡 Recommendation:")
    if elapsed_single * 20 < 0.1:
        print("  ✅ Current Numba performance is EXCELLENT (< 100ms)")
        print("  ✅ Rust migration has LOW priority")
        print("  💡 Consider Rust only if:")
        print("     - Need < 10ms latency")
        print("     - Processing 1000s of portfolios")
        print("     - Memory constraints")
    elif elapsed_single * 20 < 1.0:
        print("  ⚠️  Current performance is ACCEPTABLE (< 1s)")
        print("  💡 Rust could provide 5-10x speedup")
        print("  📊 ROI: Moderate - consider if latency matters")
    else:
        print("  ❌ Current performance NEEDS IMPROVEMENT")
        print("  🚀 Rust migration HIGHLY RECOMMENDED")
        print("  📊 ROI: High - 5-10x speedup expected")
    
    print(f"\n🔧 Optimization suggestions before Rust:")
    print(f"  1. Reduce max_iter if possible")
    print(f"  2. Use better initial guesses for mu_2")
    print(f"  3. Implement early stopping")
    print(f"  4. Cache repeated calculations")

if __name__ == "__main__":
    main()
