"""
Dataset Selector using semantic search to match queries to datasets.
"""
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dataset_manager import DatasetManager
from context_layer import ContextLayer
from ingredient_categorizer import IngredientCategorizer


class DatasetSelector:
    """selects the most relevant dataset for a given query using semantic search."""
    
    def __init__(self, dataset_manager: DatasetManager, context_layer: Optional[ContextLayer] = None,
                 ingredient_categorizer: Optional[IngredientCategorizer] = None):
        """
        init the dataset selector.
        
        Args:
            dataset_manager: DatasetManager instance with loaded datasets
            context_layer: Optional ContextLayer instance for custom semantic descriptions
            ingredient_categorizer: Optional IngredientCategorizer for ingredient-aware search
        """
        self.dataset_manager = dataset_manager
        self.context_layer = context_layer or ContextLayer()
        self.ingredient_categorizer = ingredient_categorizer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast model
        self.dataset_embeddings: dict = {}
        self._build_dataset_embeddings()
    
    def _build_dataset_embeddings(self):
        """build embeddings for each dataset based on its metadata."""
        for dataset_name, metadata in self.dataset_manager.dataset_metadata.items():
            # Use ContextLayer to build semantic description (uses custom if available, otherwise auto-generated)
            description = self.context_layer.build_semantic_description(dataset_name, metadata)
            
            # Generate embedding for the description
            embedding = self.model.encode(description, convert_to_numpy=True)
            self.dataset_embeddings[dataset_name] = {
                'embedding': embedding,
                'description': description,
                'has_custom_context': self.context_layer.has_context(dataset_name)
            }
    
    def select_dataset(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Select the most relevant dataset(s) for a given query.
        Uses ingredient categorization if available to enhance search.
        
        Args:
            query: Natural language query
            top_k: Number of top datasets to return
            
        Returns:
            List of tuples (dataset_name, similarity_score)
        """
        if not self.dataset_embeddings:
            return []
        
        # Extract dataset embeddings as numpy arrays
        dataset_embeddings_dict = {
            name: data['embedding'] 
            for name, data in self.dataset_embeddings.items()
        }
        
        # Use ingredient categorizer if available
        if self.ingredient_categorizer:
            similarities = self.ingredient_categorizer.similarity_search_with_ingredient(
                query, dataset_embeddings_dict, top_k=top_k
            )
        else:
            # Fall back to standard similarity search
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # similarity scores
            similarities = []
            for dataset_name, embedding in dataset_embeddings_dict.items():
                # cos similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((dataset_name, float(similarity)))
            
            # sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:top_k]
        
        return similarities
    
    def get_dataset_description(self, dataset_name: str) -> Optional[str]:
        """get the description used for a dataset."""
        if dataset_name in self.dataset_embeddings:
            return self.dataset_embeddings[dataset_name]['description']
        return None

