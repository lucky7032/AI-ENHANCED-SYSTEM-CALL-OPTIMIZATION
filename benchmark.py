import psutil
import time
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import os
from collections import defaultdict

class SystemCallBenchmark:
    def __init__(self, output_dir: str = 'benchmark_results'):
        self.output_dir = output_dir
        self.results = defaultdict(list)
        self.metrics = defaultdict(list)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'benchmark.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def measure_syscall_performance(self, syscall: Dict, iterations: int = 1000) -> Dict:
        """Measure performance of a system call"""
        metrics = {
            'execution_time': [],
            'cpu_usage': [],
            'memory_usage': [],
            'io_counters': []
        }
        
        process = psutil.Process()
        
        for _ in range(iterations):
            # Record start metrics
            start_time = time.time()
            start_cpu = process.cpu_percent()
            start_memory = process.memory_percent()
            start_io = process.io_counters()
            
            # Execute system call (simulated)
            self._simulate_syscall(syscall)
            
            # Record end metrics
            end_time = time.time()
            end_cpu = process.cpu_percent()
            end_memory = process.memory_percent()
            end_io = process.io_counters()
            
            # Calculate metrics
            metrics['execution_time'].append(end_time - start_time)
            metrics['cpu_usage'].append((start_cpu + end_cpu) / 2)
            metrics['memory_usage'].append((start_memory + end_memory) / 2)
            metrics['io_counters'].append({
                'read_bytes': end_io.read_bytes - start_io.read_bytes,
                'write_bytes': end_io.write_bytes - start_io.write_bytes
            })
        
        return self._aggregate_metrics(metrics)
    
    def _simulate_syscall(self, syscall: Dict):
        """Simulate a system call execution"""
        # This is a simplified simulation
        # In a real implementation, you would execute actual system calls
        time.sleep(0.001)  # Simulate some work
    
    def _aggregate_metrics(self, metrics: Dict) -> Dict:
        """Aggregate performance metrics"""
        aggregated = {}
        
        for key, values in metrics.items():
            if key == 'io_counters':
                aggregated[key] = {
                    'read_bytes': np.mean([v['read_bytes'] for v in values]),
                    'write_bytes': np.mean([v['write_bytes'] for v in values])
                }
            else:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated
    
    def compare_optimizations(self, original_syscall: Dict, optimized_syscall: Dict):
        """Compare performance between original and optimized system calls"""
        self.logger.info("Starting performance comparison...")
        
        # Measure original performance
        original_metrics = self.measure_syscall_performance(original_syscall)
        self.results['original'].append(original_metrics)
        
        # Measure optimized performance
        optimized_metrics = self.measure_syscall_performance(optimized_syscall)
        self.results['optimized'].append(optimized_metrics)
        
        # Calculate improvements
        improvements = self._calculate_improvements(original_metrics, optimized_metrics)
        self.metrics['improvements'].append(improvements)
        
        self.logger.info(f"Performance improvements: {improvements}")
    
    def _calculate_improvements(self, original: Dict, optimized: Dict) -> Dict:
        """Calculate performance improvements"""
        improvements = {}
        
        for key in ['execution_time', 'cpu_usage', 'memory_usage']:
            original_mean = original[key]['mean']
            optimized_mean = optimized[key]['mean']
            improvement = ((original_mean - optimized_mean) / original_mean) * 100
            improvements[key] = improvement
        
        return improvements
    
    def generate_report(self):
        """Generate a comprehensive benchmark report"""
        self.logger.info("Generating benchmark report...")
        
        # Create summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results['original']),
            'average_improvements': self._calculate_average_improvements(),
            'detailed_results': self._get_detailed_results()
        }
        
        # Save summary to JSON
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Generate visualizations
        self._generate_visualizations()
        
        self.logger.info("Benchmark report generated successfully!")
    
    def _calculate_average_improvements(self) -> Dict:
        """Calculate average performance improvements"""
        improvements = self.metrics['improvements']
        if not improvements:
            return {}
        
        return {
            metric: np.mean([imp[metric] for imp in improvements])
            for metric in improvements[0].keys()
        }
    
    def _get_detailed_results(self) -> Dict:
        """Get detailed results for all tests"""
        return {
            'original': self.results['original'],
            'optimized': self.results['optimized'],
            'improvements': self.metrics['improvements']
        }
    
    def _generate_visualizations(self):
        """Generate performance visualization plots"""
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('System Call Optimization Benchmark Results')
        
        # Plot execution time comparison
        self._plot_metric_comparison(axes[0, 0], 'execution_time', 'Execution Time (s)')
        
        # Plot CPU usage comparison
        self._plot_metric_comparison(axes[0, 1], 'cpu_usage', 'CPU Usage (%)')
        
        # Plot memory usage comparison
        self._plot_metric_comparison(axes[1, 0], 'memory_usage', 'Memory Usage (%)')
        
        # Plot improvements
        self._plot_improvements(axes[1, 1])
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'benchmark_results.png'))
        plt.close()
    
    def _plot_metric_comparison(self, ax, metric: str, title: str):
        """Plot comparison of a specific metric"""
        original = [r[metric]['mean'] for r in self.results['original']]
        optimized = [r[metric]['mean'] for r in self.results['optimized']]
        
        x = np.arange(len(original))
        width = 0.35
        
        ax.bar(x - width/2, original, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, optimized, width, label='Optimized', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Value')
        ax.legend()
    
    def _plot_improvements(self, ax):
        """Plot performance improvements"""
        improvements = self.metrics['improvements']
        if not improvements:
            return
        
        metrics = list(improvements[0].keys())
        values = {metric: [imp[metric] for imp in improvements] for metric in metrics}
        
        x = np.arange(len(metrics))
        ax.bar(x, [np.mean(values[metric]) for metric in metrics])
        
        ax.set_title('Average Performance Improvements')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)

def main():
    # Initialize benchmark
    benchmark = SystemCallBenchmark()
    
    try:
        # Example system calls to test
        original_syscall = {
            'pid': 1234,
            'name': 'test_process',
            'syscall': 'read',
            'timestamp': time.time()
        }
        
        optimized_syscall = {
            'pid': 1234,
            'name': 'test_process',
            'syscall': 'read',
            'timestamp': time.time(),
            'optimized': True,
            'buffer_size': 8192,
            'write_mode': 'buffered'
        }
        
        # Run benchmark
        benchmark.compare_optimizations(original_syscall, optimized_syscall)
        
        # Generate report
        benchmark.generate_report()
        
    except Exception as e:
        benchmark.logger.error(f"Error during benchmarking: {str(e)}")
        raise

if __name__ == "__main__":
    main() 