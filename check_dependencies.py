import subprocess
import re
import sys
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

def main():
    # Get required and installed packages
    required = get_required_packages()
    installed = get_installed_packages()
    
    # Check which packages are missing or have wrong version
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
                    installed_correctly.append(f"{package} (required: {req_version}, installed: {inst_version})")
                else:
                    wrong_version.append(f"{package} (required: {req_version}, installed: {inst_version})")
            elif parse_version(inst_version) != parse_version(req_version):
                wrong_version.append(f"{package} (required: {req_version}, installed: {inst_version})")
            else:
                installed_correctly.append(f"{package} (required: {req_version}, installed: {inst_version})")
    
    # Print results
    print("\n=== DEPENDENCY CHECK RESULTS ===\n")
    
    print(f"Total required packages: {len(required)}")
    print(f"Correctly installed: {len(installed_correctly)}")
    print(f"Missing: {len(missing)}")
    print(f"Wrong version: {len(wrong_version)}")
    
    if installed_correctly:
        print("\n✅ CORRECTLY INSTALLED PACKAGES:")
        for pkg in sorted(installed_correctly):
            print(f"  - {pkg}")
    
    if missing:
        print("\n❌ MISSING PACKAGES:")
        for pkg in sorted(missing):
            print(f"  - {pkg} (required: {required[pkg]})")
    
    if wrong_version:
        print("\n⚠️ PACKAGES WITH VERSION MISMATCH:")
        for pkg in sorted(wrong_version):
            print(f"  - {pkg}")
    
    # Suggest installation commands for missing packages
    if missing or wrong_version:
        print("\n--- INSTALLATION SUGGESTIONS ---")
        
        if len(missing) <= 10:
            missing_str = " ".join([f"{pkg}=={required[pkg]}" for pkg in missing])
            if missing_str:
                print(f"\nTo install missing packages:")
                print(f"pip install {missing_str}")
        else:
            print("\nToo many missing packages. Run install_packages.py to install all dependencies.")
            
        # Special handling for ta-lib which often needs manual installation
        if 'ta-lib' in missing:
            print("\nNOTE: ta-lib requires special installation:")
            print("Visit: https://github.com/mrjbq7/ta-lib for installation instructions")
            
        # Special handling for packages that might need conda
        conda_packages = ['tensorflow', 'torch', 'scipy', 'numpy']
        missing_conda = [pkg for pkg in missing if pkg in conda_packages]
        if missing_conda:
            print("\nSome packages might be better installed with conda:")
            for pkg in missing_conda:
                print(f"conda install -c conda-forge {pkg}={required[pkg]}")

if __name__ == "__main__":
    main() 