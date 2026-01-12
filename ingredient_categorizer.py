"""
LLM-Powered Ingredient Categorizer.
Uses LLM to extract ingredient names from queries and deduce potential ingredient types.
"""
import os
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from dataset_manager import DatasetManager
from context_layer import ContextLayer
import json


class IngredientCategorizer:
    """
    Uses LLM to extract ingredient names from queries and deduce potential types.
    Embeds this information for enhanced similarity search.
    """
    
    def __init__(self, dataset_manager: DatasetManager, context_layer: Optional[ContextLayer] = None,
                 llm_provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the LLM-powered ingredient categorizer.
        
        Args:
            dataset_manager: DatasetManager instance
            context_layer: Optional ContextLayer instance
            llm_provider: LLM provider to use ("openai" or "anthropic")
            api_key: API key for LLM provider (if None, reads from environment)
        """
        self.dataset_manager = dataset_manager
        self.context_layer = context_layer or ContextLayer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_provider = llm_provider
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        # Get available dataset names and types for LLM context
        self.available_datasets = dataset_manager.list_datasets()
        self.dataset_types = self._build_dataset_type_list()
        
        # Cache for ingredient extractions and categorizations
        self.extraction_cache: Dict[str, str] = {}
        self.categorization_cache: Dict[str, List[str]] = {}
    
    def _build_dataset_type_list(self) -> List[str]:
        """Build a list of potential ingredient types from dataset names and contexts."""
        types = set()
        
        # Extract types from dataset names
        type_mappings = {
            'surfactants': 'surfactant',
            'neurotoxin': 'neurotoxin',
            'neurodevelopment': 'neurodevelopmental_toxin',
            'neurotoxinspubmed': 'neurotoxin',
            'iarc1': 'carcinogen_group1',
            'iarc2a': 'carcinogen_group2a',
            'iarc2b': 'carcinogen_group2b',
            'possiblycarcin': 'possible_carcinogen',
            'extremelyHazardous': 'extremely_hazardous',
            'bisphenols': 'bisphenol',
            'mycotoxin': 'mycotoxin',
            'aminoacids': 'amino_acid',
            'flavornet': 'flavor_compound',
            'ghs': 'ghs_classified',
            'atsdr': 'atsdr_substance',
        }
        
        for dataset_name in self.available_datasets:
            if dataset_name in type_mappings:
                types.add(type_mappings[dataset_name])
            else:
                # Try to get from context layer
                context = self.context_layer.get_context(dataset_name)
                if context and context.get('domain'):
                    types.add(context['domain'])
                # Also use dataset name as a potential type
                types.add(dataset_name.replace('_', ' ').replace('-', ' '))
        
        return sorted(list(types))
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200) -> str:
        """
        Call LLM with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens for response (default: 200, use higher for longer responses)
            
        Returns:
            LLM response text
        """
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        if self.llm_provider == "openai":
            return self._call_openai(prompt, system_prompt, max_tokens)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            system_msg = system_prompt or ""
            
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_msg,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Error calling Anthropic API: {e}")
    
    def extract_ingredient_name(self, query: str, use_cache: bool = True) -> Optional[str]:
        """
        Use LLM to extract ingredient name from query.
        
        Args:
            query: User query
            use_cache: Whether to use cached results
            
        Returns:
            Extracted ingredient name, or None if not found
        """
        if use_cache and query.lower() in self.extraction_cache:
            cached = self.extraction_cache[query.lower()]
            return cached if cached != "None" else None
        
        system_prompt = (
            "You are a helpful assistant that extracts chemical ingredient names from user queries. "
            "Return only the ingredient name, or 'None' if no ingredient is mentioned. "
            "Be concise - return just the ingredient name without any explanation."
        )
        
        prompt = f"""Extract the chemical ingredient name from this query. If no ingredient is mentioned, return 'None'.

Query: {query}

Ingredient name:"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            ingredient = response.strip()
            
            # Clean up response
            if ingredient.lower() in ['none', 'null', 'n/a', '']:
                ingredient = None
            else:
                # Remove quotes if present
                ingredient = ingredient.strip('"\'')
            
            # Cache result
            if use_cache:
                self.extraction_cache[query.lower()] = ingredient if ingredient else "None"
            
            return ingredient
        except Exception as e:
            print(f"Warning: Error extracting ingredient name: {e}")
            return None
    
    def deduce_ingredient_types(self, ingredient_name: str, use_cache: bool = True) -> List[str]:
        """
        Use LLM to deduce potential ingredient types for an ingredient.
        
        Args:
            ingredient_name: Name of the ingredient
            use_cache: Whether to use cached results
            
        Returns:
            List of potential ingredient types
        """
        if use_cache and ingredient_name.lower() in self.categorization_cache:
            return self.categorization_cache[ingredient_name.lower()]
        
        # Build context about available dataset types
        dataset_info = "\n".join([f"- {dt}" for dt in self.dataset_types[:20]])  # Limit to first 20
        
        system_prompt = (
            "You are a chemical classification expert. Based on an ingredient name, "
            "determine what types of chemical categories it might belong to. "
            "Return a JSON array of potential types from the provided list."
        )
        
        prompt = f"""Given the ingredient name "{ingredient_name}", determine which of these chemical types it might belong to.

Available types:
{dataset_info}

Return a JSON array of the most likely types (2-4 types). Examples: ["surfactant", "neurotoxin"] or ["carcinogen_group1", "extremely_hazardous"]

Types (JSON array):"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            
            # Parse JSON response
            response = response.strip()
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            types = json.loads(response)
            if isinstance(types, list):
                # Filter to only include types that match our available types
                valid_types = [t for t in types if any(
                    dt.lower().replace('_', ' ').replace('-', ' ') == t.lower().replace('_', ' ').replace('-', ' ')
                    or t.lower() in dt.lower() or dt.lower() in t.lower()
                    for dt in self.dataset_types
                )]
                
                # Cache result
                if use_cache:
                    self.categorization_cache[ingredient_name.lower()] = valid_types
                
                return valid_types
            else:
                return []
        except Exception as e:
            print(f"Warning: Error deducing ingredient types: {e}")
            # Fallback: return empty list
            return []
    
    def build_enriched_query(self, query: str, ingredient_name: Optional[str], 
                            ingredient_types: List[str]) -> str:
        """
        Build an enriched query string with ingredient and type information.
        
        Args:
            query: Original query
            ingredient_name: Extracted ingredient name
            ingredient_types: Deduced ingredient types
            
        Returns:
            Enriched query string
        """
        parts = [query]
        
        if ingredient_name and ingredient_types:
            type_str = ", ".join(ingredient_types)
            parts.append(f"Ingredient: {ingredient_name}. Potential types: {type_str}")
        
        return ". ".join(parts)
    
    def enhance_query_with_ingredient(self, query: str) -> Tuple[str, np.ndarray]:
        """
        Enhance a query by extracting ingredient name and deducing types using LLM.
        
        Args:
            query: Original query string
            
        Returns:
            Tuple of (enriched_query, embedding)
        """
        # Step 1: Extract ingredient name using LLM
        ingredient_name = self.extract_ingredient_name(query)
        
        # Step 2: If ingredient found, deduce types using LLM
        ingredient_types = []
        if ingredient_name:
            ingredient_types = self.deduce_ingredient_types(ingredient_name)
        
        # Step 3: Build enriched query
        enriched_query = self.build_enriched_query(query, ingredient_name, ingredient_types)
        
        # Step 4: Embed the enriched query
        embedding = self.embedding_model.encode(enriched_query, convert_to_numpy=True)
        
        return enriched_query, embedding
    
    def similarity_search_with_ingredient(
        self, 
        query: str, 
        dataset_embeddings: Dict[str, np.ndarray],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search enhanced with LLM-powered ingredient categorization.
        
        Args:
            query: Query string
            dataset_embeddings: Dictionary mapping dataset names to their embeddings
            top_k: Number of top results to return
            
        Returns:
            List of tuples (dataset_name, similarity_score)
        """
        # Enhance query with ingredient information
        enriched_query, query_embedding = self.enhance_query_with_ingredient(query)
        
        # Calculate similarities
        similarities = []
        for dataset_name, dataset_embedding in dataset_embeddings.items():
            similarity = np.dot(query_embedding, dataset_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(dataset_embedding)
            )
            similarities.append((dataset_name, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_ingredient_analysis(self, query: str) -> Dict[str, any]:
        """
        Get full analysis of ingredient extraction and categorization for a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with analysis results
        """
        ingredient_name = self.extract_ingredient_name(query)
        ingredient_types = []
        
        if ingredient_name:
            ingredient_types = self.deduce_ingredient_types(ingredient_name)
        
        enriched_query, embedding = self.enhance_query_with_ingredient(query)
        
        return {
            'query': query,
            'ingredient_name': ingredient_name,
            'ingredient_types': ingredient_types,
            'enriched_query': enriched_query,
            'has_ingredient': ingredient_name is not None
        }
    
    def generate_answer_from_results(self, query: str, results: Dict[str, any]) -> str:
        """
        Generate an LLM answer to the query using the dataset results.
        
        Args:
            query: Original user query
            results: Dictionary with query results (from query_system.query())
            
        Returns:
            LLM-generated answer string
        """
        system_prompt = (
            "You are a helpful assistant that provides information about chemical ingredients "
            "based on EPA datasets. Answer the user's question accurately using the provided "
            "dataset results. Be concise but informative, and cite specific information from the results."
        )
        
        # Build context from results
        context_parts = []
        
        # Add dataset information
        dataset_name = results.get('dataset') or results.get('selected_dataset', 'dataset')
        context_parts.append(f"Dataset: {dataset_name}")
        
        # Add column information
        columns = results.get('columns', [])
        if columns:
            context_parts.append(f"Available columns: {', '.join(columns)}")
        
        # Add results data
        result_data = results.get('results', [])
        if result_data:
            context_parts.append(f"\nFound {len(result_data)} result(s):")
            
            # Include all results as context (limit to first 5 to avoid token limits)
            for i, result in enumerate(result_data[:5], 1):
                context_parts.append(f"\nResult {i}:")
                for key, value in result.items():
                    # Truncate very long values
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    context_parts.append(f"  {key}: {value_str}")
        else:
            context_parts.append("\nNo matching results found in the dataset.")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Answer the user's question using the provided dataset results.

User Question: {query}

Dataset Results:
{context}

Provide a clear, informative answer to the user's question based on the results above. If the results don't contain relevant information, say so clearly."""

        try:
            # Use more tokens for answer generation (500 tokens for a comprehensive answer)
            response = self._call_llm(prompt, system_prompt, max_tokens=500)
            return response.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def clear_cache(self):
        """Clear all caches."""
        self.extraction_cache.clear()
        self.categorization_cache.clear()

