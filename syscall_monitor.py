import psutil
import json
import time
from datetime import datetime
import os
import signal
import sys
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from dataclasses import dataclass, asdict
import zlib
from pathlib import Path
import hashlib

@dataclass
class SystemCall:
    timestamp: float
    pid: int
    ppid: int
    syscall: str
    cpu_percent: float
    memory_percent: float
    io_counters: Dict[str, int]
    process_name: str
    command_line: str
    thread_count: int
    status: str
    priority: int
    nice: int
    num_fds: int
    num_threads: int
    num_ctx_switches: Dict[str, int]
    num_handles: int
    username: str
    create_time: float
    hash: str = ""

    def __post_init__(self):
        # Create a unique hash for the system call
        data = f"{self.timestamp}{self.pid}{self.syscall}{self.process_name}"
        self.hash = hashlib.sha256(data.encode()).hexdigest()

class SystemCallMonitor:
    def __init__(self, log_file: str = 'syscalls.json', 
                 buffer_size: int = 1000,
                 compression_level: int = 6,
                 max_workers: int = 4):
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.compression_level = compression_level
        self.max_workers = max_workers
        self.buffer = []
        self.process_cache = {}
        self.syscall_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        self.running = False
        self.log_queue = queue.Queue()
        self.compression_queue = queue.Queue()
        
        # Set up logging with rotation
        self._setup_logging()
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _setup_logging(self):
        """Set up logging with rotation and compression"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Set up file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/monitor.log',
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
        self.logger = logging.getLogger('SystemCallMonitor')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
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
    
    def _compress_data(self, data: str) -> bytes:
        """Compress data using zlib"""
        return zlib.compress(data.encode(), level=self.compression_level)
    
    def _write_to_file(self, data: bytes):
        """Write compressed data to file"""
        try:
            with open(self.log_file, 'ab') as f:
                f.write(data + b'\n')
        except IOError as e:
            self.logger.error(f"Error writing to file: {str(e)}")
    
    def _process_buffer(self):
        """Process and write buffered data"""
        while self.running:
            try:
                # Get data from compression queue
                data = self.compression_queue.get(timeout=1)
                
                # Compress and write data
                compressed_data = self._compress_data(data)
                self._write_to_file(compressed_data)
                
                self.compression_queue.task_done()
            except queue.Empty:
                continue
    
    def _cleanup_cache(self):
        """Clean up process cache periodically"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            # Remove dead processes from cache
            dead_pids = [pid for pid in self.process_cache 
                        if not psutil.pid_exists(pid)]
            for pid in dead_pids:
                del self.process_cache[pid]
            
            self.last_cleanup = current_time
            self.logger.debug(f"Cleaned up {len(dead_pids)} dead processes from cache")
    
    def _update_stats(self, syscall: SystemCall):
        """Update system call statistics"""
        self.syscall_stats[syscall.syscall]['count'] += 1
        self.syscall_stats[syscall.syscall]['total_time'] += time.time() - syscall.timestamp
    
    def _log_syscall(self, syscall: SystemCall):
        """Log system call with metadata"""
        try:
            # Convert system call to JSON
            syscall_dict = asdict(syscall)
            
            # Add to buffer
            self.buffer.append(syscall_dict)
            
            # Update statistics
            self._update_stats(syscall)
            
            # If buffer is full, process it
            if len(self.buffer) >= self.buffer_size:
                self._process_buffer_batch()
                
        except Exception as e:
            self.logger.error(f"Error logging system call: {str(e)}")
    
    def _process_buffer_batch(self):
        """Process a batch of buffered system calls"""
        if not self.buffer:
            return
        
        try:
            # Convert buffer to JSON string
            data = json.dumps(self.buffer)
            
            # Add to compression queue
            self.compression_queue.put(data)
            
            # Clear buffer
            self.buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error processing buffer batch: {str(e)}")
    
    def monitor_syscalls(self):
        """Main monitoring loop with improved performance"""
        self.running = True
        self.logger.info("Starting system call monitoring...")
        
        # Start compression thread
        compression_thread = threading.Thread(
            target=self._process_buffer,
            daemon=True
        )
        compression_thread.start()
        
        try:
            while self.running:
                # Clean up cache periodically
                self._cleanup_cache()
                
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
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {str(e)}")
        finally:
            # Process remaining buffer
            self._process_buffer_batch()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Wait for compression thread
            compression_thread.join()
            
            self.logger.info("System call monitoring stopped")
    
    def _process_single_process(self, process: psutil.Process):
        """Process a single process with error handling"""
        try:
            # Get process info
            process_info = self._get_process_info(process)
            if not process_info:
                return
            
            # Create system call object
            syscall = SystemCall(
                timestamp=time.time(),
                **process_info
            )
            
            # Log system call
            self._log_syscall(syscall)
            
        except Exception as e:
            self.logger.error(f"Error processing process {process.pid}: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get current monitoring statistics"""
        return {
            'syscall_stats': dict(self.syscall_stats),
            'process_cache_size': len(self.process_cache),
            'buffer_size': len(self.buffer),
            'compression_queue_size': self.compression_queue.qsize()
        }
    
    def start(self):
        """Start the monitoring process"""
        try:
            self.monitor_syscalls()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
        finally:
            self.running = False

def main():
    # Initialize monitor
    monitor = SystemCallMonitor()
    
    try:
        # Start monitoring
        monitor.start()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 