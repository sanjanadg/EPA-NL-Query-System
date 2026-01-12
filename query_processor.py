"""
Query Processor for extracting information from datasets based on natural language queries.
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from dataset_manager import DatasetManager
from sentence_transformers import SentenceTransformer
import numpy as np


class QueryProcessor:
    """Processes natural language queries to extract information from datasets."""
    
    def __init__(self, dataset_manager: DatasetManager):
        """
        Init the query processor.
        
        Args:
            dataset_manager: DatasetManager instance
        """
        self.dataset_manager = dataset_manager
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_query(self, query: str, dataset_name: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Process a natural language query against a specific dataset.
        
        Args:
            query: Natural language query
            dataset_name: Name of the dataset to query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with results and metadata
        """
        df = self.dataset_manager.get_dataset(dataset_name)
        if df is None:
            return {
                'success': False,
                'error': f'Dataset "{dataset_name}" not found',
                'results': []
            }
        
        # try multiple search strategies
        results = self._search_dataset(df, query, max_results)
        
        return {
            'success': True,
            'dataset': dataset_name,
            'query': query,
            'num_results': len(results),
            'results': results.to_dict('records') if len(results) > 0 else [],
            'columns': list(df.columns)
        }
    
    def _search_dataset(self, df: pd.DataFrame, query: str, max_results: int) -> pd.DataFrame:
        """
        Search a dataset using multiple strategies.
        
        Args:
            df: DataFrame to search
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            DataFrame with matching rows
        """
        query_lower = query.lower()
        results = []
        
        # Strat 1: direct text matching in string columns
        text_mask = pd.Series([False] * len(df))
        for col in df.columns:
            if df[col].dtype == 'object':
                text_mask |= df[col].astype(str).str.lower().str.contains(
                    query_lower, na=False, regex=False
                )
        
        if text_mask.any():
            text_results = df[text_mask]
            results.append(text_results)
        
        # Strat 2: Semantic search on text columns (if dataset is small enough)
        if len(df) < 10000:  # Only for smaller datasets due to performance
            semantic_results = self._semantic_search(df, query, max_results)
            if len(semantic_results) > 0:
                results.append(semantic_results)
        
        # Combine and deduplicate results
        if results:
            combined = pd.concat(results, ignore_index=True)
            combined = combined.drop_duplicates()
            return combined.head(max_results)
        
        # If no matches, return sample rows
        return df.head(max_results)
    
    def _semantic_search(self, df: pd.DataFrame, query: str, max_results: int) -> pd.DataFrame:
        """perform semantic search on text columns."""
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Combine text from all string columns for each row
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if not text_columns:
            return pd.DataFrame()
        
        similarities = []
        for idx, row in df.iterrows():
            row_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
            if row_text:
                row_embedding = self.model.encode(row_text, convert_to_numpy=True)
                similarity = np.dot(query_embedding, row_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(row_embedding)
                )
                similarities.append((idx, similarity))
        
        if not similarities:
            return pd.DataFrame()
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:max_results]]
        return df.loc[top_indices]


