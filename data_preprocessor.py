import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import json
import os
from collections import defaultdict
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
import gc
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
from sklearn.feature_selection import SelectKBest, f_classif

class SystemCallPreprocessor:
    def __init__(self, chunk_size: int = 100000, max_workers: int = 4):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.processed_data = None
        self.feature_columns = []
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_raw_data(self, log_file: str = 'syscalls.json') -> pd.DataFrame:
        """Load raw system call data efficiently using Dask"""
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log file {log_file} not found")
        
        self.logger.info(f"Loading data from {log_file}")
        
        # Use Dask for efficient loading of large files
        ddf = dd.read_json(log_file, lines=True)
        df = ddf.compute()
        
        self.logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data with improved outlier detection and handling"""
        self.logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_len - len(df)} duplicate records")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna()
        self.logger.info(f"Removed {initial_len - len(df)} records with missing values")
        
        # Enhanced outlier detection using multiple methods
        for col in ['cpu_percent', 'memory_percent']:
            if col in df.columns:
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                iqr_mask = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
                
                # Z-score method
                z_scores = stats.zscore(df[col])
                z_mask = abs(z_scores) < 3
                
                # Combine both methods
                df = df[iqr_mask & z_mask]
        
        self.logger.info(f"Final dataset size: {len(df)} records")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced features from the system call data"""
        self.logger.info("Starting feature extraction...")
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Process frequency features with parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            process_freq = executor.submit(lambda: df['pid'].value_counts())
            syscall_freq = executor.submit(lambda: df['syscall'].value_counts())
        
        df['process_frequency'] = df['pid'].map(process_freq.result())
        df['syscall_frequency'] = df['syscall'].map(syscall_freq.result())
        
        # Time between calls with enhanced statistics
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['time_diff_log'] = np.log1p(df['time_diff'])
        
        # Process behavior features with parallel processing
        process_stats = df.groupby('pid').agg({
            'syscall': 'count',
            'time_diff': ['mean', 'std', 'skew', 'kurtosis', 'max', 'min'],
            'cpu_percent': ['mean', 'std', 'max'],
            'memory_percent': ['mean', 'std', 'max']
        }).reset_index()
        
        # Flatten column names
        process_stats.columns = ['pid'] + [f'{col}_{agg}' for col, agg in process_stats.columns[1:]]
        df = df.merge(process_stats, on='pid', how='left')
        
        # Enhanced rolling statistics with multiple windows
        window_sizes = [5, 10, 20, 50]
        for window in window_sizes:
            df[f'rolling_mean_{window}'] = df.groupby('pid')['time_diff'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('pid')['time_diff'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'rolling_max_{window}'] = df.groupby('pid')['time_diff'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        # Enhanced system call patterns
        df['syscall_pattern'] = df.groupby('pid')['syscall'].transform(
            lambda x: x.rolling(window=3, min_periods=1).agg(lambda y: '_'.join(y))
        )
        df['pattern_length'] = df['syscall_pattern'].str.len()
        
        # Encode categorical variables
        df['syscall_encoded'] = self.label_encoder.fit_transform(df['syscall'])
        df['pattern_encoded'] = self.label_encoder.fit_transform(df['syscall_pattern'])
        
        # Store feature columns
        self.feature_columns = [
            'hour', 'minute', 'second', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos',
            'process_frequency', 'syscall_frequency',
            'time_diff', 'time_diff_log', 'total_calls',
            'time_diff_mean', 'time_diff_std', 'time_diff_skew', 'time_diff_kurtosis',
            'time_diff_max', 'time_diff_min',
            'cpu_percent_mean', 'cpu_percent_std', 'cpu_percent_max',
            'memory_percent_mean', 'memory_percent_std', 'memory_percent_max',
            'rolling_mean_5', 'rolling_std_5', 'rolling_max_5',
            'rolling_mean_10', 'rolling_std_10', 'rolling_max_10',
            'rolling_mean_20', 'rolling_std_20', 'rolling_max_20',
            'rolling_mean_50', 'rolling_std_50', 'rolling_max_50',
            'pattern_length', 'syscall_encoded', 'pattern_encoded'
        ]
        
        self.logger.info(f"Extracted {len(self.feature_columns)} features")
        return df
    
    def select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Select most important features using statistical tests"""
        self.logger.info("Selecting most important features...")
        
        # Fit feature selector
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(selected_features)} features")
        return X_selected, selected_features
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features with improved handling"""
        self.logger.info("Normalizing features...")
        
        # Identify numerical features
        numerical_features = [
            col for col in self.feature_columns
            if col not in ['syscall_encoded', 'pattern_encoded', 'is_weekend']
        ]
        
        # Normalize in chunks to handle large datasets
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            df.iloc[i:i + self.chunk_size, df.columns.get_indexer(numerical_features)] = \
                self.scaler.fit_transform(chunk[numerical_features])
            
            # Force garbage collection after each chunk
            gc.collect()
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series prediction with memory optimization"""
        self.logger.info("Preparing sequences...")
        
        sequences = []
        targets = []
        
        # Process each process ID separately to manage memory
        for pid in df['pid'].unique():
            pid_data = df[df['pid'] == pid].sort_values('timestamp')
            
            # Process in chunks
            for i in range(0, len(pid_data) - sequence_length, self.chunk_size):
                chunk = pid_data.iloc[i:i + self.chunk_size + sequence_length]
                
                for j in range(len(chunk) - sequence_length):
                    sequence = chunk[self.feature_columns].iloc[j:j+sequence_length].values
                    target = chunk['syscall_encoded'].iloc[j+sequence_length]
                    
                    sequences.append(sequence)
                    targets.append(target)
            
            # Force garbage collection after each process
            gc.collect()
        
        self.logger.info(f"Created {len(sequences)} sequences")
        return np.array(sequences), np.array(targets)
    
    def process_data(self, log_file: str = 'syscalls.json', sequence_length: int = 10) -> Dict:
        """Complete data processing pipeline with improved memory management"""
        try:
            # Load and clean data
            df = self.load_raw_data(log_file)
            df = self.clean_data(df)
            
            # Extract and normalize features
            df = self.extract_features(df)
            df = self.normalize_features(df)
            
            # Prepare sequences
            X, y = self.prepare_sequences(df, sequence_length)
            
            # Select most important features
            X_selected, selected_features = self.select_features(X, y)
            
            # Store processed data
            self.processed_data = {
                'X': X_selected,
                'y': y,
                'feature_columns': selected_features,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector
            }
            
            self.logger.info("Data processing completed successfully!")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"Error during data processing: {str(e)}")
            raise
    
    def save_processed_data(self, output_file: str = 'processed_data.npz'):
        """Save processed data with compression"""
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        self.logger.info(f"Saving processed data to {output_file}")
        
        # Save with compression
        np.savez_compressed(
            output_file,
            X=self.processed_data['X'],
            y=self.processed_data['y'],
            feature_columns=self.processed_data['feature_columns']
        )
        
        self.logger.info("Data saved successfully!")

def main():
    # Initialize preprocessor
    preprocessor = SystemCallPreprocessor()
    
    try:
        # Process the data
        processed_data = preprocessor.process_data()
        
        # Save processed data
        preprocessor.save_processed_data()
        
        print("Data preprocessing completed successfully!")
        print(f"Number of sequences: {len(processed_data['X'])}")
        print(f"Number of features: {len(processed_data['feature_columns'])}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main() 