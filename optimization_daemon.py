import psutil
import time
import json
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Set
from collections import defaultdict
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from dataclasses import dataclass, asdict
import zlib
from pathlib import Path
import hashlib
from syscall_predictor import SystemCallPredictor
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

@dataclass
class OptimizationStats:
    total_syscalls: int = 0
    optimized_syscalls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prediction_hits: int = 0
    prediction_misses: int = 0
    avg_optimization_time: float = 0.0
    total_optimization_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

class SystemCallOptimizer:
    def __init__(self, 
                 cache_size: int = 10000,
                 prediction_threshold: float = 0.8,
                 max_workers: int = 4,
                 log_dir: str = 'logs',
                 model_path: str = 'models/best_model.h5'):
        self.cache_size = cache_size
        self.prediction_threshold = prediction_threshold
        self.max_workers = max_workers
        self.log_dir = Path(log_dir)
        self.model_path = model_path
        
        # Initialize components
        self.syscall_cache = {}
        self.optimization_stats = OptimizationStats(start_time=time.time())
        self.prediction_queue = queue.Queue()
        self.optimization_queue = queue.Queue()
        self.running = False
        
        # Initialize predictor
        self.predictor = SystemCallPredictor()
        self.predictor.load_model(model_path)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _setup_logging(self):
        """Set up logging with rotation"""
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / 'optimizer.log'
        
        # Set up file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        
        # Set up formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set up logger
        self.logger = logging.getLogger('SystemCallOptimizer')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def _optimize_syscall(self, syscall: Dict) -> Dict:
        """Optimize a system call with improved strategies"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(syscall)
            if cache_key in self.syscall_cache:
                self.optimization_stats.cache_hits += 1
                return self.syscall_cache[cache_key]
            
            self.optimization_stats.cache_misses += 1
            
            # Get prediction for next syscall
            sequence = self._prepare_sequence(syscall)
            predicted_syscall, confidence = self.predictor.predict_next_syscall(sequence)
            
            if confidence >= self.prediction_threshold:
                self.optimization_stats.prediction_hits += 1
                optimized_syscall = self._apply_prediction_based_optimization(
                    syscall, predicted_syscall
                )
            else:
                self.optimization_stats.prediction_misses += 1
                optimized_syscall = self._default_optimization(syscall)
            
            # Update cache
            self._update_cache(cache_key, optimized_syscall)
            
            # Update statistics
            optimization_time = time.time() - start_time
            self.optimization_stats.total_optimization_time += optimization_time
            self.optimization_stats.optimized_syscalls += 1
            self.optimization_stats.avg_optimization_time = (
                self.optimization_stats.total_optimization_time / 
                self.optimization_stats.optimized_syscalls
            )
            
            return optimized_syscall
            
        except Exception as e:
            self.logger.error(f"Error optimizing syscall: {str(e)}")
            return self._default_optimization(syscall)
    
    def _get_cache_key(self, syscall: Dict) -> str:
        """Generate a unique cache key for a system call"""
        key_data = f"{syscall['pid']}_{syscall['syscall']}_{syscall['timestamp']}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: Dict):
        """Update cache with LRU policy"""
        if len(self.syscall_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.syscall_cache))
            del self.syscall_cache[oldest_key]
        
        self.syscall_cache[key] = value
    
    def _prepare_sequence(self, syscall: Dict) -> np.ndarray:
        """Prepare input sequence for prediction"""
        # Extract features
        features = [
            syscall['cpu_percent'],
            syscall['memory_percent'],
            syscall['thread_count'],
            syscall['num_fds'],
            syscall['num_handles']
        ]
        
        # Scale features
        features_scaled = self.scaler.fit_transform([features])
        
        # Reshape for LSTM input
        return features_scaled.reshape(1, -1, len(features))
    
    def _apply_prediction_based_optimization(self, 
                                          syscall: Dict, 
                                          predicted_syscall: str) -> Dict:
        """Apply optimization based on prediction"""
        optimized_syscall = syscall.copy()
        
        # Pre-allocate resources based on prediction
        if predicted_syscall in ['read', 'write']:
            optimized_syscall['buffer_size'] = self._optimize_buffer_size(syscall)
        
        # Set process priority based on prediction
        if predicted_syscall in ['fork', 'exec']:
            optimized_syscall['priority'] = self._optimize_process_priority(syscall)
        
        # Optimize thread count based on prediction
        if predicted_syscall in ['pthread_create', 'pthread_join']:
            optimized_syscall['thread_count'] = self._optimize_thread_count(syscall)
        
        return optimized_syscall
    
    def _default_optimization(self, syscall: Dict) -> Dict:
        """Apply default optimization strategies"""
        optimized_syscall = syscall.copy()
        
        # Basic optimizations
        optimized_syscall['buffer_size'] = self._optimize_buffer_size(syscall)
        optimized_syscall['priority'] = self._optimize_process_priority(syscall)
        optimized_syscall['thread_count'] = self._optimize_thread_count(syscall)
        
        return optimized_syscall
    
    def _optimize_buffer_size(self, syscall: Dict) -> int:
        """Optimize buffer size based on system memory"""
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        
        # Calculate optimal buffer size based on available memory
        optimal_size = min(
            int(available_memory * 0.1),  # Use 10% of available memory
            1024 * 1024  # Max 1MB
        )
        
        return max(optimal_size, 4096)  # Minimum 4KB
    
    def _optimize_process_priority(self, syscall: Dict) -> int:
        """Optimize process priority based on system load"""
        cpu_percent = psutil.cpu_percent()
        
        if cpu_percent > 80:
            return 10  # High priority
        elif cpu_percent > 50:
            return 0   # Normal priority
        else:
            return -10  # Low priority
    
    def _optimize_thread_count(self, syscall: Dict) -> int:
        """Optimize thread count based on CPU cores"""
        cpu_count = psutil.cpu_count()
        current_threads = syscall['thread_count']
        
        # Adjust thread count based on CPU usage
        if psutil.cpu_percent() > 80:
            return max(1, current_threads - 1)  # Reduce threads
        else:
            return min(cpu_count * 2, current_threads + 1)  # Increase threads
    
    def _process_predictions(self):
        """Process predictions from queue"""
        while self.running:
            try:
                # Get prediction from queue
                prediction = self.prediction_queue.get(timeout=1)
                
                # Process prediction
                self._handle_prediction(prediction)
                
                self.prediction_queue.task_done()
            except queue.Empty:
                continue
    
    def _handle_prediction(self, prediction: Dict):
        """Handle a prediction result"""
        try:
            syscall = prediction['syscall']
            predicted_syscall = prediction['predicted_syscall']
            confidence = prediction['confidence']
            
            if confidence >= self.prediction_threshold:
                # Apply prediction-based optimization
                optimized_syscall = self._apply_prediction_based_optimization(
                    syscall, predicted_syscall
                )
                
                # Add to optimization queue
                self.optimization_queue.put(optimized_syscall)
                
        except Exception as e:
            self.logger.error(f"Error handling prediction: {str(e)}")
    
    def optimize_syscalls(self):
        """Main optimization loop with improved performance"""
        self.running = True
        self.logger.info("Starting system call optimization...")
        
        # Start prediction processing thread
        prediction_thread = threading.Thread(
            target=self._process_predictions,
            daemon=True
        )
        prediction_thread.start()
        
        try:
            while self.running:
                # Get all processes
                processes = psutil.process_iter(['pid', 'name', 'cmdline'])
                
                # Process each process in parallel
                futures = []
                for process in processes:
                    future = self.executor.submit(
                        self._process_single_process,
                        process
                    )
                    futures.append(future)
                
                # Wait for all processes to be processed
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing process: {str(e)}")
                
                # Update statistics
                self._update_statistics()
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in optimization loop: {str(e)}")
        finally:
            # Wait for prediction thread
            prediction_thread.join()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            self.logger.info("System call optimization stopped")
    
    def _process_single_process(self, process: psutil.Process):
        """Process a single process with error handling"""
        try:
            # Get process info
            process_info = self._get_process_info(process)
            if not process_info:
                return
            
            # Optimize syscall
            optimized_syscall = self._optimize_syscall(process_info)
            
            # Add to optimization queue
            self.optimization_queue.put(optimized_syscall)
            
        except Exception as e:
            self.logger.error(f"Error processing process {process.pid}: {str(e)}")
    
    def _get_process_info(self, process: psutil.Process) -> Dict:
        """Get detailed process information with error handling"""
        try:
            return {
                'pid': process.pid,
                'ppid': process.ppid(),
                'name': process.name(),
                'cmdline': process.cmdline(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'io_counters': process.io_counters()._asdict(),
                'thread_count': process.num_threads(),
                'status': process.status(),
                'priority': process.nice(),
                'num_fds': process.num_fds(),
                'num_ctx_switches': process.num_ctx_switches()._asdict(),
                'num_handles': process.num_handles(),
                'username': process.username(),
                'create_time': process.create_time()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            self.logger.warning(f"Error getting process info: {str(e)}")
            return None
    
    def _update_statistics(self):
        """Update optimization statistics"""
        self.optimization_stats.memory_usage = psutil.Process().memory_percent()
        self.optimization_stats.cpu_usage = psutil.Process().cpu_percent()
        self.optimization_stats.end_time = time.time()
    
    def get_stats(self) -> Dict:
        """Get current optimization statistics"""
        return asdict(self.optimization_stats)
    
    def start(self):
        """Start the optimization process"""
        try:
            self.optimize_syscalls()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
        finally:
            self.running = False

def main():
    # Initialize optimizer
    optimizer = SystemCallOptimizer()
    
    try:
        # Start optimization
        optimizer.start()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 