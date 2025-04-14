import os
import re
import shutil

# Files/folders to preserve
PRESERVED = [
    'stucute.txt',
    'create_empty_structure.py',
    '.env',
    'main.py',
]

def parse_structure(filename):
    """Parse stucute.txt to extract directories and files"""
    directories = set()
    files = set()
    
    # Manual parsing to get the structure since regex is challenging
    # This is a simplified approach for this specific file
    manual_dirs = [
        'core', 'analysis', 'ml', 'execution', 'strategies', 'utils', 'web',
        'interfaces', 'security', 'deployment', 'optimization', 'research',
        'tests', 'docs', 'examples', 'scripts', 'config'
    ]
    
    # Add manual directories
    for d in manual_dirs:
        directories.add(d)
    
    # Add common files from the structure
    common_files = [
        '.gitignore', 'README.md', 'requirements.txt', 'setup.py',
        'LICENSE', 'CHANGELOG.md'
    ]
    
    for f in common_files:
        files.add(f)
    
    # Add subdirectories
    sub_dirs = {
        'core': ['__init__.py'],
        'analysis': ['__init__.py', 'technical', 'fundamental', 'sentiment', 'statistical', 
                    'volatility', 'options', 'portfolio', 'risk', 'market_structure', 'advanced'],
        'ml': ['__init__.py', 'neural_networks', 'reinforcement_learning', 'generative', 
              'nlp', 'advanced_ml', 'data_science', 'infrastructure'],
        'execution': ['__init__.py', 'order_management', 'execution_algorithms', 
                     'position_management', 'trade_lifecycle', 'safety'],
        'strategies': ['__init__.py', 'momentum', 'mean_reversion', 'volatility', 
                      'market_making', 'arbitrage', 'event_driven', 'algorithmic', 'crypto_specific'],
        'utils': ['__init__.py', 'data', 'visualization', 'validation', 'simulation', 'system', 'crypto'],
        'web': ['__init__.py', 'server', 'services', 'static', 'templates'],
        'interfaces': ['__init__.py', 'exchanges', 'data_providers', 'services'],
        'security': ['__init__.py'],
        'deployment': ['__init__.py', 'docker', 'kubernetes', 'cloud'],
        'optimization': ['__init__.py', 'hyperparameter', 'strategy', 'system'],
        'research': ['__init__.py', 'market_research', 'strategy_research', 'model_research', 'news_research'],
        'tests': ['__init__.py', 'unit', 'integration', 'system'],
        'docs': ['images'],
        'examples': ['__init__.py'],
        'scripts': [],
        'config': []
    }
    
    # Process subdirectories
    for parent, subs in sub_dirs.items():
        for sub in subs:
            if sub.endswith('.py'):
                # This is a file
                files.add(f"{parent}/{sub}")
            else:
                # This is a directory
                directories.add(f"{parent}/{sub}")
                
                # Add __init__.py to subdirectories
                if not sub.startswith('__') and parent != 'docs' and sub not in ['static', 'templates', 'images']:
                    files.add(f"{parent}/{sub}/__init__.py")
    
    # Add web static and templates subdirectories
    web_dirs = {
        'web/static': ['css', 'js', 'img'],
        'web/static/js': ['core', 'components', 'visualizations', 'utils'],
        'web/templates': ['trading', 'analytics', 'ai', 'settings']
    }
    
    for parent, subs in web_dirs.items():
        for sub in subs:
            directories.add(f"{parent}/{sub}")
    
    # Add deeper level directories
    deep_dirs = {
        'analysis/technical': ['__init__.py', 'indicators.py', 'oscillators.py', 'chart_patterns.py', 
                            'fibonacci.py', 'elliot_wave.py', 'ichimoku.py', 'divergence.py', 'custom_indicators.py'],
        'analysis/fundamental': ['__init__.py', 'token_economics.py', 'network_metrics.py', 'developer_activity.py',
                              'adoption_metrics.py', 'regulatory_analysis.py', 'competitor_analysis.py', 'valuation_models.py'],
        'analysis/sentiment': ['__init__.py', 'news_analysis.py', 'social_media.py', 'fear_greed_index.py',
                             'influencer_tracking.py', 'forum_analysis.py', 'search_trends.py', 'crowd_psychology.py'],
        'execution/order_management': ['__init__.py', 'order_types.py', 'smart_routing.py', 'limit_orders.py',
                                    'iceberg_orders.py', 'conditional_orders.py', 'order_book_placement.py', 'order_monitoring.py'],
        'strategies/momentum': ['__init__.py', 'trend_following.py', 'breakout.py', 'momentum_factor.py', 'acceleration.py'],
        'utils/data': ['__init__.py', 'preprocessing.py', 'normalization.py', 'augmentation.py', 'feature_engineering.py',
                     'time_series_utils.py', 'data_validation.py', 'data_pipeline.py'],
        'ml/neural_networks': ['__init__.py', 'lstm.py', 'gru.py', 'cnn.py', 'attention.py', 'transformers.py',
                             'neural_ode.py', 'graph_networks.py', 'autoencoder.py']
    }
    
    for parent, items in deep_dirs.items():
        for item in items:
            if item.endswith('.py'):
                files.add(f"{parent}/{item}")
            else:
                directories.add(f"{parent}/{item}")
                files.add(f"{parent}/{item}/__init__.py")
    
    return directories, files

def create_structure(directories, files):
    """Create all directories and empty files"""
    print("Creating directories...")
    for directory in sorted(directories):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nCreating empty files...")
    for file in sorted(files):
        directory = os.path.dirname(file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Create empty file
        with open(file, 'w') as f:
            pass
        print(f"Created file: {file}")

def clean_filesystem(valid_paths):
    """Remove files and directories not in the structure"""
    # Get all files and directories in the current directory
    all_items = os.listdir('.')
    
    for item in all_items:
        # Skip preserved items
        if item in PRESERVED or item.startswith('.'):
            continue
            
        # Check if item is in our valid paths
        if item not in valid_paths and item != 'LICENSE':
            if os.path.isdir(item):
                print(f"Removing directory: {item}")
                shutil.rmtree(item)
            else:
                print(f"Removing file: {item}")
                os.remove(item)

if __name__ == "__main__":
    print("Generating directory/file structure...")
    directories, files = parse_structure('stucute.txt')
    
    # Create sets of top-level items for cleaning
    top_level_items = {d.split('/')[0] for d in directories}
    top_level_items.update({f.split('/')[0] for f in files})
    
    print(f"Found {len(directories)} directories and {len(files)} files to create")
    print(f"Top-level directories: {', '.join(sorted(top_level_items))}")
    
    # Remove LICENSE directory if it exists
    if os.path.isdir('LICENSE'):
        print("Removing LICENSE directory...")
        shutil.rmtree('LICENSE')
    
    # Clean existing filesystem
    print("\nRemoving items not in the structure...")
    clean_filesystem(top_level_items)
    
    # Create the structure
    create_structure(directories, files)
    
    print("\nStructure creation complete!") 