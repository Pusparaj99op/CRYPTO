import os
import re

def extract_paths():
    """
    Extract all file and directory paths from the structure file.
    Returns a list of entries to create.
    """
    structure_entries = []
    
    # Hard-code main directories and subdirectories
    dirs = [
        "All CRYPTO",
        "All CRYPTO/core",
        "All CRYPTO/analysis",
        "All CRYPTO/analysis/technical",
        "All CRYPTO/analysis/fundamental",
        "All CRYPTO/analysis/sentiment",
        "All CRYPTO/analysis/statistical",
        "All CRYPTO/analysis/volatility",
        "All CRYPTO/analysis/options",
        "All CRYPTO/analysis/portfolio",
        "All CRYPTO/analysis/risk",
        "All CRYPTO/analysis/market_structure",
        "All CRYPTO/analysis/advanced",
        "All CRYPTO/ml",
        "All CRYPTO/ml/neural_networks",
        "All CRYPTO/ml/reinforcement_learning",
        "All CRYPTO/ml/generative",
        "All CRYPTO/ml/nlp",
        "All CRYPTO/ml/advanced_ml",
        "All CRYPTO/ml/data_science",
        "All CRYPTO/ml/infrastructure",
        "All CRYPTO/execution",
        "All CRYPTO/execution/order_management",
        "All CRYPTO/execution/execution_algorithms",
        "All CRYPTO/execution/position_management",
        "All CRYPTO/execution/trade_lifecycle",
        "All CRYPTO/execution/safety",
        "All CRYPTO/strategies",
        "All CRYPTO/strategies/momentum",
        "All CRYPTO/strategies/mean_reversion",
        "All CRYPTO/strategies/volatility",
        "All CRYPTO/strategies/market_making",
        "All CRYPTO/strategies/arbitrage",
        "All CRYPTO/strategies/event_driven",
        "All CRYPTO/strategies/algorithmic",
        "All CRYPTO/strategies/crypto_specific",
        "All CRYPTO/utils",
        "All CRYPTO/utils/data",
        "All CRYPTO/utils/visualization",
        "All CRYPTO/utils/validation",
        "All CRYPTO/utils/simulation",
        "All CRYPTO/utils/system",
        "All CRYPTO/utils/crypto",
        "All CRYPTO/web",
        "All CRYPTO/web/server",
        "All CRYPTO/web/services",
        "All CRYPTO/web/static",
        "All CRYPTO/web/static/css",
        "All CRYPTO/web/static/js",
        "All CRYPTO/web/static/js/core",
        "All CRYPTO/web/static/js/components",
        "All CRYPTO/web/static/js/visualizations",
        "All CRYPTO/web/static/js/utils",
        "All CRYPTO/web/static/img",
        "All CRYPTO/web/templates",
        "All CRYPTO/web/templates/trading",
        "All CRYPTO/web/templates/analytics",
        "All CRYPTO/web/templates/ai",
        "All CRYPTO/web/templates/settings",
        "All CRYPTO/interfaces",
        "All CRYPTO/interfaces/exchanges",
        "All CRYPTO/interfaces/data_providers",
        "All CRYPTO/interfaces/services",
        "All CRYPTO/security",
        "All CRYPTO/deployment",
        "All CRYPTO/deployment/docker",
        "All CRYPTO/deployment/kubernetes",
        "All CRYPTO/deployment/cloud",
        "All CRYPTO/deployment/cloud/terraform",
        "All CRYPTO/optimization",
        "All CRYPTO/optimization/hyperparameter",
        "All CRYPTO/optimization/strategy",
        "All CRYPTO/optimization/system",
        "All CRYPTO/research",
        "All CRYPTO/research/market_research",
        "All CRYPTO/research/strategy_research",
        "All CRYPTO/research/model_research",
        "All CRYPTO/research/news_research",
        "All CRYPTO/tests",
        "All CRYPTO/tests/unit",
        "All CRYPTO/tests/integration",
        "All CRYPTO/tests/system",
        "All CRYPTO/docs",
        "All CRYPTO/docs/images",
        "All CRYPTO/examples",
        "All CRYPTO/scripts",
        "All CRYPTO/config"
    ]
    
    # Add all directories
    for dir_path in dirs:
        structure_entries.append((dir_path, True))  # True = directory
    
    # Read the structure file to extract file entries
    with open('structure.txt', 'r') as f:
        content = f.read()
    
    # Extract file entries using regex
    file_pattern = r'(?:├──|└──)\s+([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)'
    file_matches = re.findall(file_pattern, content)
    
    # Map to parent directories
    parent_dirs = {
        # Core dir files
        "__init__.py": "All CRYPTO/core",
        "ai_orchestrator.py": "All CRYPTO/core",
        "binance_client.py": "All CRYPTO/core",
        "data_hub.py": "All CRYPTO/core",
        "knowledge_graph.py": "All CRYPTO/core",
        "config_manager.py": "All CRYPTO/core",
        "system_monitor.py": "All CRYPTO/core",
        "quantum_simulator.py": "All CRYPTO/core",
        
        # Root files
        ".gitignore": "All CRYPTO",
        "README.md": "All CRYPTO",
        "requirements.txt": "All CRYPTO",
        "setup.py": "All CRYPTO",
        "structure.txt": "All CRYPTO",
        "LICENSE": "All CRYPTO",
        "CHANGELOG.md": "All CRYPTO",
    }
    
    # Add file entries where parent is known
    for file_name in file_matches:
        file_name = file_name.strip()
        if file_name in parent_dirs:
            full_path = os.path.join(parent_dirs[file_name], file_name)
            structure_entries.append((full_path, False))  # False = file
    
    # Add __init__.py to all Python packages
    for dir_path in dirs:
        if not dir_path.endswith(("css", "js", "img", "images", "cloud/terraform")):
            init_path = os.path.join(dir_path, "__init__.py")
            structure_entries.append((init_path, False))
    
    return structure_entries

def create_structure(entries):
    """Create all directories and files."""
    created_dirs = []
    created_files = []
    
    # First create all directories
    for path, is_dir in entries:
        if is_dir:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                created_dirs.append(path)
                print(f"Created directory: {path}")
    
    # Then create all files
    for path, is_dir in entries:
        if not is_dir:
            if not os.path.exists(path):
                # Ensure parent directory exists
                parent_dir = os.path.dirname(path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    if parent_dir not in created_dirs:
                        created_dirs.append(parent_dir)
                
                # Create the file
                with open(path, 'w') as f:
                    pass  # Create empty file
                created_files.append(path)
                print(f"Created file: {path}")
    
    return created_dirs, created_files

def main():
    """Main function to create the directory structure."""
    if not os.path.exists('structure.txt'):
        print("Error: structure.txt not found in the current directory.")
        return
    
    print("Extracting paths from structure.txt...")
    entries = extract_paths()
    print(f"Found {len(entries)} entries to create ({len([e for e in entries if e[1]])} directories, {len([e for e in entries if not e[1]])} files)")
    
    print("\nCreating directories and files...")
    created_dirs, created_files = create_structure(entries)
    
    print(f"\nSummary: Created {len(created_dirs)} directories and {len(created_files)} files.")

if __name__ == "__main__":
    main() 