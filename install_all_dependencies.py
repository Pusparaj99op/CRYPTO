import subprocess
import sys
import re
import os
import time
from pkg_resources import parse_version

def get_required_packages():
    """Parse requirements.txt to get required packages and versions"""
    required = {}
    with open('requirements.txt', 'r') as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('python '):
                continue
                
            # Parse package name and version
            match = re.match(r'^([a-zA-Z0-9_.-]+)==([0-9a-zA-Z.-]+)$', line)
            if match:
                package, version = match.groups()
                required[package.lower()] = version
    
    return required

def get_installed_packages():
    """Get dictionary of installed packages and their versions"""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'], 
                           capture_output=True, text=True)
    
    installed = {}
    for line in result.stdout.split('\n'):
        if '==' in line:
            package, version = line.split('==', 1)
            installed[package.lower()] = version
    
    return installed

def check_dependencies():
    """Check which packages are missing or have wrong version"""
    required = get_required_packages()
    installed = get_installed_packages()
    
    missing = []
    wrong_version = []
    installed_correctly = []
    
    for package, req_version in required.items():
        if package not in installed:
            missing.append(package)
        else:
            inst_version = installed[package]
            # Check if versions match (ignoring patch version)
            req_parts = req_version.split('.')
            inst_parts = inst_version.split('.')
            
            # Consider only major and minor versions for comparison
            if len(req_parts) >= 2 and len(inst_parts) >= 2:
                if req_parts[0] == inst_parts[0] and req_parts[1] == inst_parts[1]:
                    installed_correctly.append((package, req_version, inst_version))
                else:
                    wrong_version.append((package, req_version, inst_version))
            elif parse_version(inst_version) != parse_version(req_version):
                wrong_version.append((package, req_version, inst_version))
            else:
                installed_correctly.append((package, req_version, inst_version))
    
    return {
        'required': required,
        'installed': installed,
        'missing': missing,
        'wrong_version': wrong_version,
        'installed_correctly': installed_correctly
    }

def run_pip_install(packages):
    """Install packages using pip"""
    cmd = f"{sys.executable} -m pip install {packages}"
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0

def run_conda_install(packages, channel="conda-forge"):
    """Install packages using conda"""
    cmd = f"conda install -c {channel} {packages} -y"
    print(f"Running: {cmd}")
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0

def display_results(results):
    """Display dependency check results"""
    print("\n=== DEPENDENCY CHECK RESULTS ===\n")
    
    print(f"Total required packages: {len(results['required'])}")
    print(f"Correctly installed: {len(results['installed_correctly'])}")
    print(f"Missing: {len(results['missing'])}")
    print(f"Wrong version: {len(results['wrong_version'])}")
    
    if results['installed_correctly']:
        print("\n✅ CORRECTLY INSTALLED PACKAGES:")
        for pkg, req_ver, inst_ver in sorted(results['installed_correctly']):
            print(f"  - {pkg} (required: {req_ver}, installed: {inst_ver})")
    
    if results['missing']:
        print("\n❌ MISSING PACKAGES:")
        for pkg in sorted(results['missing']):
            print(f"  - {pkg} (required: {results['required'][pkg]})")
    
    if results['wrong_version']:
        print("\n⚠️ PACKAGES WITH VERSION MISMATCH:")
        for pkg, req_ver, inst_ver in sorted(results['wrong_version']):
            print(f"  - {pkg} (required: {req_ver}, installed: {inst_ver})")

def main():
    # Check current state
    print("Checking current dependencies...")
    results = check_dependencies()
    display_results(results)
    
    if not results['missing'] and not results['wrong_version']:
        print("\n🎉 All dependencies are correctly installed!")
        return
    
    # Install dependencies
    print("\n=== INSTALLING MISSING DEPENDENCIES ===\n")
    
    # Define package groups with preferred installation method
    package_groups = [
        # Special packages with specific handling
        {
            'name': 'Critical dependencies',
            'packages': ['typing-extensions==4.5.0'],
            'method': 'pip'
        },
        {
            'name': 'Core scientific',
            'packages': ['numpy', 'pandas', 'scikit-learn', 'scipy', 'statsmodels', 'matplotlib', 'seaborn'],
            'method': 'conda',
            'channel': 'conda-forge'
        },
        {
            'name': 'Machine learning',
            'packages': ['torch', 'xgboost', 'optuna'],
            'method': 'conda',
            'channel': 'conda-forge'
        },
        {
            'name': 'Web framework',
            'packages': ['flask', 'dash', 'plotly', 'flask-socketio', 'flask-restful', 'flask-sqlalchemy'],
            'method': 'pip'
        },
        {
            'name': 'Database',
            'packages': ['sqlalchemy', 'pymongo', 'redis'],
            'method': 'pip'
        },
        {
            'name': 'Crypto APIs',
            'packages': ['python-binance', 'ccxt', 'pycoingecko', 'websocket-client'],
            'method': 'pip'
        },
        {
            'name': 'Technical analysis',
            'packages': ['ta', 'pandas-ta'],
            'method': 'pip'
        },
        {
            'name': 'NLP',
            'packages': ['nltk', 'textblob', 'vadersentiment'],
            'method': 'pip'
        },
        {
            'name': 'Utilities',
            'packages': ['tqdm', 'joblib', 'python-dotenv', 'loguru', 'pydantic', 'fastapi', 'uvicorn'],
            'method': 'pip'
        },
        {
            'name': 'Testing',
            'packages': ['pytest', 'pytest-mock'],
            'method': 'pip'
        },
        {
            'name': 'Ollama',
            'packages': ['ollama'],
            'method': 'pip'
        }
    ]
    
    # Process each package group
    for group in package_groups:
        # Filter to only missing or wrong version packages
        to_install = []
        for pkg in group['packages']:
            pkg_lower = pkg.lower().split('==')[0]
            if pkg_lower in results['missing'] or any(pkg_lower == p[0] for p in results['wrong_version']):
                # Add version if it's in the required list
                if '==' not in pkg and pkg_lower in results['required']:
                    pkg = f"{pkg}=={results['required'][pkg_lower]}"
                to_install.append(pkg)
        
        if not to_install:
            continue
            
        print(f"\nInstalling {group['name']} packages...")
        pkg_str = " ".join(to_install)
        
        if group['method'] == 'conda':
            channel = group.get('channel', 'conda-forge')
            run_conda_install(pkg_str, channel)
        else:
            run_pip_install(pkg_str)
            
        time.sleep(1)  # Small delay to prevent overwhelming output
    
    # Special handling for ta-lib
    if 'ta-lib' in results['missing']:
        print("\n=== SPECIAL HANDLING FOR TA-LIB ===")
        print("TA-Lib requires special installation:")
        print("1. Download and install C library: https://github.com/mrjbq7/ta-lib/releases")
        print("2. Then run: pip install ta-lib")
        
        choice = input("\nDo you want to try installing TA-Lib with pip? (y/n): ")
        if choice.lower() == 'y':
            run_pip_install('ta-lib')
    
    # Check dependencies again after installation
    print("\n=== CHECKING DEPENDENCIES AFTER INSTALLATION ===")
    final_results = check_dependencies()
    display_results(final_results)
    
    if not final_results['missing'] and not final_results['wrong_version']:
        print("\n🎉 All dependencies are correctly installed!")
    else:
        print("\n⚠️ Some dependencies are still missing or have version mismatches.")
        print("You may need to install them manually.")

if __name__ == "__main__":
    main() 