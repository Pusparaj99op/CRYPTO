import subprocess
import sys
import time

def run_command(command):
    """Run a pip install command and print the result"""
    print(f"\n>>> Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if process.returncode == 0:
        print("Installation successful!")
    else:
        print(f"Error during installation: {process.stderr}")
    
    # Return True if successful, False otherwise
    return process.returncode == 0

def main():
    # Define package groups
    package_groups = [
        # Core scientific packages
        "numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 scipy==1.11.3 statsmodels==0.14.0",
        
        # Basic visualization and utilities
        "matplotlib==3.7.2 seaborn==0.12.2 tqdm==4.66.1 joblib==1.3.2 python-dotenv==1.0.0 loguru==0.7.0",
        
        # Explicitly set typing-extensions version first (critical for compatibility)
        "typing-extensions==4.5.0",
        
        # Database-related
        "sqlalchemy==2.0.20 pymongo==4.5.0 redis==4.6.0",
        
        # Web framework
        "flask==2.2.5 dash==2.13.0 plotly==5.16.1",
        
        # Web extensions
        "flask-socketio==5.3.4 flask-restful==0.3.10 flask-sqlalchemy==3.0.5",
        
        # API libraries
        "requests==2.31.0 websocket-client==1.6.1",
        
        # Crypto-specific APIs
        "python-binance==1.0.17 ccxt==3.1.54 pycoingecko==3.1.0",
        
        # Technical analysis
        "ta==0.10.2 pandas-ta==0.3.14b0",
        
        # NLP and text processing
        "nltk==3.8.1 textblob==0.17.1 vaderSentiment==3.3.2",
        
        # API framework
        "pydantic==1.10.8 fastapi==0.95.2 uvicorn==0.22.0",
        
        # Testing
        "pytest==7.4.0 pytest-mock==3.11.1",
        
        # Ollama
        "ollama==0.1.6",
        
        # New packages
        "transformers==4.30.2 pytorch-lightning==2.0.6"
    ]
    
    # Install each package group
    for i, packages in enumerate(package_groups):
        print(f"\n[{i+1}/{len(package_groups)}] Installing: {packages}")
        success = run_command(f"{sys.executable} -m pip install {packages}")
        
        # If installation fails, print warning but continue
        if not success:
            print(f"Warning: Failed to install some packages in group {i+1}. Continuing with next group.")
        
        # Small delay to prevent overwhelming the console
        time.sleep(1)
    
    print("\n\nNote: Some packages require special installation procedures:")
    print("1. TA-Lib: Follow instructions at https://github.com/mrjbq7/ta-lib")
    print("2. PyTorch: Visit https://pytorch.org/get-started/locally/ to get the correct command for your system")
    print("3. TensorFlow: If installation failed, you may need to use a specific version for your system")

if __name__ == "__main__":
    print("Starting installation of packages for BNB Future AI Trading System")
    main()
    print("\nInstallation process completed!") 