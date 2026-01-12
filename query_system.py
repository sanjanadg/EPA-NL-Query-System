"""
Main Natural Language Query System for EPA Datasets.
"""
from typing import Dict, Any, Optional
from dataset_manager import DatasetManager
from dataset_selector import DatasetSelector
from query_processor import QueryProcessor
from context_layer import ContextLayer
from ingredient_categorizer import IngredientCategorizer
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class EPAQuerySystem:
    """Main system for querying EPA datasets using natural language."""
    
    def __init__(self, data_directory: str = "data", context_config_file: str = "dataset_contexts.json"):
        """
        Initialize the query system.
        
        Args:
            data_directory: Path to directory containing JSON files
            context_config_file: Path to JSON file containing dataset context definitions
        """
        print("Initializing EPA Query System...")
        
        # Initialize context layer first
        print("Loading context-aware layer...")
        self.context_layer = ContextLayer(context_config_file)
        
        # Initialize components
        self.dataset_manager = DatasetManager(data_directory)
        print("Loading datasets...")
        self.dataset_manager.load_datasets()
        
        if not self.dataset_manager.datasets:
            print("Warning: No datasets loaded. Please add JSON files to the data directory.")
        else:
            print(f"Loaded {len(self.dataset_manager.datasets)} dataset(s)")
        
        print("Initializing LLM-powered ingredient categorizer...")
        # Initialize ingredient categorizer (will use API key from environment)
        try:
            self.ingredient_categorizer = IngredientCategorizer(
                self.dataset_manager, 
                self.context_layer,
                llm_provider="openai"  # Can be changed to "anthropic"
            )
            print("  ✓ Ingredient categorizer ready")
        except Exception as e:
            print(f"  ⚠ Warning: Ingredient categorizer not available: {e}")
            print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable to enable")
            self.ingredient_categorizer = None
        
        print("Initializing dataset selector with context layer...")
        self.dataset_selector = DatasetSelector(
            self.dataset_manager, 
            self.context_layer,
            self.ingredient_categorizer
        )
        
        print("Initializing query processor...")
        self.query_processor = QueryProcessor(self.dataset_manager)
        
        print("System ready!")
    
    def query(self, query: str, max_results: int = 10, 
              dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: Natural language query string
            max_results: Maximum number of results to return
            dataset_name: Optional specific dataset to query (if None, auto-selects)
            
        Returns:
            Dictionary with query results and metadata
        """
        if not self.dataset_manager.datasets:
            return {
                'success': False,
                'error': 'No datasets available. Please add JSON files to the data directory.',
                'query': query
            }
        
        # If dataset not specified, select the most relevant one
        if dataset_name is None:
            # Show ingredient analysis if categorizer is available
            if self.ingredient_categorizer:
                analysis = self.ingredient_categorizer.get_ingredient_analysis(query)
                if analysis.get('has_ingredient'):
                    print(f"Detected ingredient: {analysis['ingredient_name']}")
                    if analysis.get('ingredient_types'):
                        print(f"  Potential types: {', '.join(analysis['ingredient_types'])}")
            
            selected_datasets = self.dataset_selector.select_dataset(query, top_k=1)
            if not selected_datasets:
                return {
                    'success': False,
                    'error': 'Could not match query to any dataset',
                    'query': query
                }
            
            dataset_name, similarity_score = selected_datasets[0]
            print(f"Selected dataset: {dataset_name} (similarity: {similarity_score:.3f})")
        else:
            if dataset_name not in self.dataset_manager.datasets:
                return {
                    'success': False,
                    'error': f'Dataset "{dataset_name}" not found',
                    'query': query,
                    'available_datasets': self.dataset_manager.list_datasets()
                }
            similarity_score = None
        
        # Process the query against the selected dataset
        result = self.query_processor.process_query(query, dataset_name, max_results)
        
        # Add selection metadata
        if similarity_score is not None:
            result['selected_dataset'] = dataset_name
            result['selection_confidence'] = similarity_score
        
        return result
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all available datasets with their metadata."""
        datasets_info = {}
        for name in self.dataset_manager.list_datasets():
            metadata = self.dataset_manager.get_dataset_info(name)
            datasets_info[name] = {
                'num_rows': metadata['num_rows'],
                'num_columns': metadata['num_columns'],
                'columns': metadata['columns']
            }
        return {
            'num_datasets': len(datasets_info),
            'datasets': datasets_info
        }
    
    def reload_datasets(self):
        """Reload all datasets from the data directory."""
        self.dataset_manager.datasets.clear()
        self.dataset_manager.dataset_metadata.clear()
        self.dataset_manager.load_datasets()
        self.context_layer.reload_contexts()
        self.dataset_selector._build_dataset_embeddings()
    
    def get_context_layer(self) -> ContextLayer:
        """Get the context layer instance for managing dataset definitions."""
        return self.context_layer
    
    def set_dataset_context(self, dataset_name: str, context: dict):
        """
        Set or update the semantic context for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            context: Dictionary with context information containing:
                - semantic_description: Detailed description of the dataset
                - keywords: List of keywords (optional)
                - example_queries: List of example queries (optional)
                - domain: Domain/category (optional)
                - column_descriptions: Dict mapping column names to descriptions (optional)
        """
        self.context_layer.set_context(dataset_name, context)
        # Rebuild embeddings to include the new context
        self.dataset_selector._build_dataset_embeddings()
    
    def generate_answer(self, query: str, results: Dict[str, Any]) -> str:
        """
        Generate an LLM answer to the query using the dataset results.
        
        Args:
            query: Original user query
            results: Dictionary with query results (from query() method)
            
        Returns:
            LLM-generated answer string
        """
        if not self.ingredient_categorizer:
            return "LLM answer generation not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        
        return self.ingredient_categorizer.generate_answer_from_results(query, results)

