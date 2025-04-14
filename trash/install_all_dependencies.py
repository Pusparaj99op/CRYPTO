import subprocess
import sys
import re
import os
import time
from pkg_resources import parse_version
import shutil

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

def read_structure_file(file_path):
    """
    Parse the structure.txt file to extract all mentioned file names
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract all file and directory names from structure.txt
    mentioned_items = set()
    
    # Add common items that are implicitly mentioned or considered part of the structure
    mentioned_items.add('structure.txt')
    mentioned_items.add('README.md')
    mentioned_items.add('LICENSE')
    mentioned_items.add('CHANGELOG.md')
    mentioned_items.add('.gitignore')
    mentioned_items.add('requirements.txt')
    mentioned_items.add('setup.py')
    mentioned_items.add('trash')  # Don't move the trash folder itself
    
    # Parse file contents
    lines = content.split('\n')
    for line in lines:
        # Remove comments
        if '#' in line:
            line = line.split('#')[0].strip()
        
        # Extract file/directory names
        if '─' in line:
            parts = line.split('─')
            if len(parts) > 1:
                # Get the name part and clean it
                name_part = parts[-1].strip()
                if name_part:
                    # Clean up the name (remove spaces, etc.)
                    clean_name = name_part.strip()
                    mentioned_items.add(clean_name)

    return mentioned_items

def get_all_files_and_dirs(root_dir='.'):
    """
    Get all files and directories in the current directory (non-recursive)
    """
    return set(os.listdir(root_dir))

def move_to_trash(item_path):
    """
    Move a file or directory to the trash folder
    """
    # Create trash folder if it doesn't exist
    trash_dir = 'trash'
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)
    
    # Move the item to trash
    destination = os.path.join(trash_dir, os.path.basename(item_path))
    
    # Handle case where destination already exists
    if os.path.exists(destination):
        # Append a number to make the name unique
        count = 1
        name, ext = os.path.splitext(os.path.basename(item_path))
        while os.path.exists(os.path.join(trash_dir, f"{name}_{count}{ext}")):
            count += 1
        destination = os.path.join(trash_dir, f"{name}_{count}{ext}")
    
    try:
        shutil.move(item_path, destination)
        print(f"Moved {item_path} to {destination}")
    except Exception as e:
        print(f"Error moving {item_path}: {e}")

def main():
    # Special directories to ignore (never move these)
    special_dirs = {'.git', 'venv', '.conda', 'trash'}
    
    # Read structure file
    mentioned_items = read_structure_file('structure.txt')
    
    # Get all files and directories
    all_items = get_all_files_and_dirs()
    
    # Find items not mentioned in structure.txt
    unmentioned_items = all_items - mentioned_items - special_dirs
    
    # Directory names mentioned in structure.txt
    directory_names = {
        'core', 'analysis', 'ml', 'execution', 'strategies', 'utils',
        'web', 'interfaces', 'security', 'deployment', 'optimization',
        'research', 'tests', 'docs', 'examples', 'scripts', 'config'
    }
    
    # Don't move top-level directories that are mentioned in structure.txt
    unmentioned_items = {item for item in unmentioned_items if item not in directory_names}
    
    # Move unmentioned items to trash
    if unmentioned_items:
        print(f"Moving {len(unmentioned_items)} items to trash folder:")
        for item in unmentioned_items:
            move_to_trash(item)
    else:
        print("No unmentioned items found to move.")

if __name__ == "__main__":
    main() 