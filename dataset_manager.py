"""
Dataset Manager for loading and managing EPA JSON datasets.
"""
import os
import pandas as pd
import json
from typing import Dict, List, Optional
from pathlib import Path


class DatasetManager:
    """manages loading and accessing multiple EPA datasets."""
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the dataset manager.
        
        Args:
            data_directory: Path to directory containing JSON files
        """
        self.data_directory = Path(data_directory)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_metadata: Dict[str, dict] = {}
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        load all JSON files from the data directory.
        
        Supports JSON files containing:
        - Arrays of objects: [{"key": "value"}, ...]
        - Single objects: {"key": "value"}
        - Objects with data arrays: {"data": [{"key": "value"}, ...]}
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        if not self.data_directory.exists():
            self.data_directory.mkdir(parents=True, exist_ok=True)
            print(f"Created data directory: {self.data_directory}")
            return {}
        
        json_files = list(self.data_directory.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.data_directory}")
            return {}
        
        for json_file in json_files:
            dataset_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    # Array of objects
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Check if it's a single object or has a data array
                    if 'data' in json_data and isinstance(json_data['data'], list):
                        # Object with data array
                        df = pd.DataFrame(json_data['data'])
                    else:
                        # Single object - convert to single-row DataFrame
                        df = pd.DataFrame([json_data])
                else:
                    print(f"Unsupported JSON structure in {json_file}")
                    continue
                
                if df.empty:
                    print(f"Warning: {dataset_name} is empty")
                    continue
                
                self.datasets[dataset_name] = df
                
                # Store metadata about the dataset
                self.dataset_metadata[dataset_name] = {
                    'file_path': str(json_file),
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'columns': list(df.columns),
                    'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
                }
                
                print(f"Loaded dataset: {dataset_name} ({len(df)} rows, {len(df.columns)} columns)")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {json_file}: {e}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return self.datasets
    
    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get a specific dataset by name."""
        return self.datasets.get(dataset_name)
    
    def get_dataset_info(self, dataset_name: str) -> Optional[dict]:
        """Get metadata about a specific dataset."""
        return self.dataset_metadata.get(dataset_name)
    
    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self.datasets.keys())
    
    def search_dataset(self, dataset_name: str, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Search within a specific dataset using text matching.
        
        Args:
            dataset_name: Name of the dataset to search
            query: Search query string
            query: Number of results to return
            
        Returns:
            DataFrame with matching rows
        """
        if dataset_name not in self.datasets:
            return pd.DataFrame()
        
        df = self.datasets[dataset_name]
        query_lower = query.lower()
        
        # Search across all string columns
        mask = pd.Series([False] * len(df))
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False)
        
        results = df[mask]
        
        # If no text matches, try numeric search
        if len(results) == 0:
            try:
                # Try to find numeric columns that might match
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if query_lower.replace('.', '').replace('-', '').isdigit():
                            value = float(query_lower)
                            # Find closest matches
                            results = df.iloc[(df[col] - value).abs().argsort()[:top_k]]
                            break
            except:
                pass
        
        return results.head(top_k)

