"""
Main entry point for the EPA Query System.
Example usage and interactive interface.
"""
import json
from query_system import EPAQuerySystem


def print_results(result: dict):
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    
    if not result.get('success', False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    print(f"Query: {result.get('query', 'N/A')}")
    print(f"Dataset: {result.get('dataset', result.get('selected_dataset', 'N/A'))}")
    
    if 'selection_confidence' in result:
        print(f"Selection Confidence: {result['selection_confidence']:.3f}")
    
    print(f"\nFound {result.get('num_results', 0)} result(s)")
    print(f"Columns: {', '.join(result.get('columns', []))}")
    
    results = result.get('results', [])
    if results:
        print("\nResults:")
        print("-" * 60)
        for i, row in enumerate(results, 1):
            print(f"\nResult {i}:")
            for key, value in row.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo matching results found.")
    
    print("="*60 + "\n")


def interactive_mode():
    """Run the system in interactive mode."""
    print("EPA Natural Language Query System")
    print("="*60)
    print("Type 'quit' or 'exit' to exit")
    print("Type 'list' to see available datasets")
    print("Type 'reload' to reload datasets")
    print("Type 'contexts' to see dataset context definitions")
    print("Type 'context <dataset>' to see context for a specific dataset")
    print("Type 'analyze <query>' to see LLM-powered ingredient extraction and categorization")
    print("="*60 + "\n")
    
    system = EPAQuerySystem()
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'list':
                datasets = system.list_datasets()
                print(f"\nAvailable datasets ({datasets['num_datasets']}):")
                context_layer = system.get_context_layer()
                for name, info in datasets['datasets'].items():
                    has_context = context_layer.has_context(name)
                    context_marker = " [has custom context]" if has_context else ""
                    print(f"  - {name}: {info['num_rows']} rows, {info['num_columns']} columns{context_marker}")
                    print(f"    Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
                continue
            
            if query.lower() == 'reload':
                print("Reloading datasets...")
                system.reload_datasets()
                print("Datasets reloaded!")
                continue
            
            if query.lower() == 'contexts':
                context_layer = system.get_context_layer()
                contexts = context_layer.list_contexts()
                print(f"\nDataset Context Definitions ({len(contexts)}):")
                print("-" * 60)
                for dataset_name, context in contexts.items():
                    print(f"\n{dataset_name}:")
                    if context.get('semantic_description'):
                        desc = context['semantic_description']
                        print(f"  Description: {desc[:150]}{'...' if len(desc) > 150 else ''}")
                    if context.get('domain'):
                        print(f"  Domain: {context['domain']}")
                    if context.get('keywords'):
                        keywords = context['keywords']
                        if isinstance(keywords, list):
                            print(f"  Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
                continue
            
            if query.lower().startswith('context '):
                dataset_name = query[8:].strip()
                context_layer = system.get_context_layer()
                context = context_layer.get_context(dataset_name)
                if context:
                    print(f"\nContext for '{dataset_name}':")
                    print(json.dumps(context, indent=2))
                else:
                    print(f"\nNo custom context defined for '{dataset_name}'")
                    print("You can define one in dataset_contexts.json")
                continue
            
            if query.lower().startswith('analyze '):
                query_to_analyze = query[8:].strip()
                if query_to_analyze:
                    print(f"\nAnalyzing query: {query_to_analyze}")
                    analysis = system.get_ingredient_analysis(query_to_analyze)
                    if 'error' in analysis:
                        print(f"  Error: {analysis['error']}")
                    else:
                        print(f"\n  Ingredient extracted: {analysis.get('ingredient_name', 'None')}")
                        if analysis.get('ingredient_types'):
                            print(f"  Potential types: {', '.join(analysis['ingredient_types'])}")
                        print(f"  Enriched query: {analysis.get('enriched_query', 'N/A')}")
                else:
                    print("Please provide a query to analyze")
                continue
            
            result = system.query(query)
            print_results(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    
    interactive_mode()

