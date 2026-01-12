"""
Utility script for managing dataset context definitions.
Helps you add, update, and view custom semantic descriptions for datasets.
"""
import json
import sys
from context_layer import ContextLayer


def print_usage():
    """Print usage information."""
    print("Dataset Context Manager")
    print("=" * 60)
    print("Usage:")
    print("  python manage_contexts.py list")
    print("  python manage_contexts.py view <dataset_name>")
    print("  python manage_contexts.py add <dataset_name>")
    print("  python manage_contexts.py remove <dataset_name>")
    print("=" * 60)

def list_contexts(context_layer: ContextLayer):
    """List all defined contexts."""
    contexts = context_layer.list_contexts()
    if not contexts:
        print("\nNo contexts defined. Use 'add' to create one.")
        return
    
    print(f"\nDefined contexts ({len(contexts)}):")
    print("-" * 60)
    for dataset_name, context in contexts.items():
        has_desc = bool(context.get('semantic_description'))
        domain = context.get('domain', 'N/A')
        print(f"  {dataset_name}: {domain} {'✓' if has_desc else '✗'}")

def view_context(context_layer: ContextLayer, dataset_name: str):
    """View context for a specific dataset."""
    context = context_layer.get_context(dataset_name)
    if context:
        print(f"\nContext for '{dataset_name}':")
        print(json.dumps(context, indent=2))
    else:
        print(f"\nNo context defined for '{dataset_name}'")
        print("\nTo add a context, use:")
        print(f"  python manage_contexts.py add {dataset_name}")


def add_context(context_layer: ContextLayer, dataset_name: str):
    """Interactively add a context for a dataset."""
    print(f"\nAdding context for '{dataset_name}'")
    print("(Press Enter to skip optional fields)\n")
    
    context = {}
    
    # Required: semantic description
    desc = input("Semantic description (required): ").strip()
    if not desc:
        print("Error: Semantic description is required!")
        return
    context['semantic_description'] = desc
    
    # Optional: keywords
    keywords_input = input("Keywords (comma-separated, optional): ").strip()
    if keywords_input:
        context['keywords'] = [k.strip() for k in keywords_input.split(',')]
    
    # Optional: domain
    domain = input("Domain/category (optional): ").strip()
    if domain:
        context['domain'] = domain
    
    # Optional: example queries
    examples_input = input("Example queries (comma-separated, optional): ").strip()
    if examples_input:
        context['example_queries'] = [q.strip() for q in examples_input.split(',')]
    
    # Save
    context_layer.set_context(dataset_name, context)
    print(f"\n✓ Context saved for '{dataset_name}'")
    print("\nSaved context:")
    print(json.dumps(context, indent=2))

def remove_context(context_layer: ContextLayer, dataset_name: str):
    """Remove context for a dataset."""
    if context_layer.has_context(dataset_name):
        confirm = input(f"Are you sure you want to remove context for '{dataset_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            context_layer.remove_context(dataset_name)
            print(f"✓ Context removed for '{dataset_name}'")
        else:
            print("Cancelled.")
    else:
        print(f"No context defined for '{dataset_name}'")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    context_layer = ContextLayer()
    
    if command == 'list':
        list_contexts(context_layer)
    
    elif command == 'view':
        if len(sys.argv) < 3:
            print("Error: Please specify a dataset name")
            print("Usage: python manage_contexts.py view <dataset_name>")
            sys.exit(1)
        view_context(context_layer, sys.argv[2])
    
    elif command == 'add':
        if len(sys.argv) < 3:
            print("Error: Please specify a dataset name")
            print("Usage: python manage_contexts.py add <dataset_name>")
            sys.exit(1)
        add_context(context_layer, sys.argv[2])
    
    elif command == 'remove':
        if len(sys.argv) < 3:
            print("Error: Please specify a dataset name")
            print("Usage: python manage_contexts.py remove <dataset_name>")
            sys.exit(1)
        remove_context(context_layer, sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()

