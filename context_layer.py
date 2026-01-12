"""
Context-Aware Layer for managing custom semantic descriptions of datasets.
Allows users to define their own semantic descriptions that connect to databases.
"""
import json
import os
from typing import Dict, Optional, List
from pathlib import Path


class ContextLayer:
    """
    Manages custom semantic descriptions for datasets.
    Allows users to define their own descriptions that connect semantic queries to databases.
    """
    
    def __init__(self, config_file: str = "dataset_contexts.json"):
        """
        Initialize the context layer.
        
        Args:
            config_file: Path to JSON file containing dataset context definitions
        """
        self.config_file = Path(config_file)
        self.contexts: Dict[str, dict] = {}
        self._load_contexts()
    
    def _load_contexts(self):
        """Load dataset contexts from configuration file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Extract dataset_contexts if it's nested, otherwise use the whole structure
                if isinstance(config, dict) and 'dataset_contexts' in config:
                    self.contexts = config['dataset_contexts']
                elif isinstance(config, dict):
                    # Assume the whole dict is the contexts if no 'dataset_contexts' key
                    self.contexts = config
                else:
                    self.contexts = {}
                print(f"Loaded {len(self.contexts)} dataset context(s) from {self.config_file}")
            except json.JSONDecodeError as e:
                print(f"Error parsing context file {self.config_file}: {e}")
                self.contexts = {}
            except Exception as e:
                print(f"Error loading context file {self.config_file}: {e}")
                self.contexts = {}
        else:
            print(f"Context file {self.config_file} not found. Creating default structure.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration file structure."""
        default_config = {
            "dataset_contexts": {},
            "description": "Define semantic descriptions for your datasets here. Each dataset can have:",
            "fields": {
                "semantic_description": "A detailed description of what the dataset contains and what queries it can answer",
                "keywords": "List of keywords that help match queries to this dataset",
                "example_queries": "Example queries that would match this dataset",
                "domain": "The domain or category this dataset belongs to (e.g., 'toxicity', 'air_quality', 'chemicals')",
                "column_descriptions": "Optional: descriptions of what each column represents"
            }
        }
        self.contexts = default_config.get("dataset_contexts", {})
        self._save_contexts(default_config)
    
    def _save_contexts(self, config: Optional[Dict] = None):
        """Save contexts to configuration file."""
        if config is None:
            config = {
                "dataset_contexts": self.contexts,
                "description": "Define semantic descriptions for your datasets here.",
                "fields": {
                    "semantic_description": "A detailed description of what the dataset contains and what queries it can answer",
                    "keywords": "List of keywords that help match queries to this dataset",
                    "example_queries": "Example queries that would match this dataset",
                    "domain": "The domain or category this dataset belongs to",
                    "column_descriptions": "Optional: descriptions of what each column represents"
                }
            }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving context file: {e}")
    
    def get_context(self, dataset_name: str) -> Optional[dict]:
        """
        Get the context definition for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with context information, or None if not found
        """
        return self.contexts.get(dataset_name)
    
    def set_context(self, dataset_name: str, context: dict):
        """
        Set or update the context definition for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            context: Dictionary with context information containing:
                - semantic_description: Detailed description of the dataset
                - keywords: List of keywords (optional)
                - example_queries: List of example queries (optional)
                - domain: Domain/category (optional)
                - column_descriptions: Dict mapping column names to descriptions (optional)
        """
        self.contexts[dataset_name] = context
        self._save_contexts()
    
    def build_semantic_description(self, dataset_name: str, metadata: dict) -> str:
        """
        Build a semantic description for a dataset using custom context if available,
        otherwise fall back to auto-generated description.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Metadata dictionary from DatasetManager
            
        Returns:
            Semantic description string
        """
        context = self.get_context(dataset_name)
        
        if context and context.get('semantic_description'):
            # Use custom semantic description
            description = context['semantic_description']
            
            # Add keywords if available
            if context.get('keywords'):
                keywords = context['keywords']
                if isinstance(keywords, list):
                    description += f" Keywords: {', '.join(keywords)}."
                else:
                    description += f" Keywords: {keywords}."
            
            # Add domain if available
            if context.get('domain'):
                description += f" Domain: {context['domain']}."
            
            # Add example queries if available
            if context.get('example_queries'):
                examples = context['example_queries']
                if isinstance(examples, list):
                    description += f" Example queries: {', '.join(examples)}."
                else:
                    description += f" Example queries: {examples}."
            
            # Add column descriptions if available
            if context.get('column_descriptions'):
                col_descs = context['column_descriptions']
                if isinstance(col_descs, dict):
                    col_info = []
                    for col, desc in col_descs.items():
                        if col in metadata.get('columns', []):
                            col_info.append(f"{col}: {desc}")
                    if col_info:
                        description += f" Column meanings: {'; '.join(col_info)}."
            
            return description
        else:
            # Fall back to auto-generated description
            return self._auto_generate_description(dataset_name, metadata)
    
    def _auto_generate_description(self, dataset_name: str, metadata: dict) -> str:
        """
        Auto-generate a description when no custom context is available.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Metadata dictionary
            
        Returns:
            Auto-generated description
        """
        columns = ', '.join(metadata.get('columns', []))
        num_rows = metadata.get('num_rows', 0)
        
        description = f"Dataset: {dataset_name}. "
        description += f"Contains {num_rows} records. "
        description += f"Columns: {columns}. "
        
        # Add sample data context if available
        sample = metadata.get('sample_data', [])
        if sample:
            description += f"Sample data includes: {str(sample[:2])}"
        
        return description
    
    def list_contexts(self) -> Dict[str, dict]:
        """List all defined contexts."""
        return self.contexts.copy()
    
    def has_context(self, dataset_name: str) -> bool:
        """Check if a dataset has a custom context defined."""
        return dataset_name in self.contexts and self.contexts[dataset_name].get('semantic_description')
    
    def remove_context(self, dataset_name: str):
        """Remove a context definition for a dataset."""
        if dataset_name in self.contexts:
            del self.contexts[dataset_name]
            self._save_contexts()
    
    def reload_contexts(self):
        """Reload contexts from the configuration file."""
        self._load_contexts()

