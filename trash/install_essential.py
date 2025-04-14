import subprocess
import sys

def install_packages(packages):
    """Install a list of packages"""
    print(f"Installing: {packages}")
    cmd = f"{sys.executable} -m pip install {packages}"
    process = subprocess.run(cmd, shell=True)
    return process.returncode == 0

def main():
    # Install packages in groups
    print("Installing essential packages...")
    
    # Install typing-extensions first as it's a critical dependency
    install_packages("typing-extensions==4.5.0")
    
    # Core ML packages
    print("\nInstalling core ML packages...")
    install_packages("torch==2.0.1 xgboost==1.7.6 optuna==3.3.0")
    
    # Core web packages
    print("\nInstalling web and API packages...")
    install_packages("flask==2.2.5 flask-socketio==5.3.4 flask-restful==0.3.10 flask-sqlalchemy==3.0.5")
    install_packages("dash==2.13.0 plotly==5.16.1")
    
    # Data sources
    print("\nInstalling data source APIs...")
    install_packages("python-binance==1.0.17 pycoingecko==3.1.0 websocket-client==1.6.1")
    
    # Technical analysis
    print("\nInstalling technical analysis packages...")
    install_packages("ta==0.10.2 pandas-ta==0.3.14b0")
    
    # Utilities
    print("\nInstalling utilities...")
    install_packages("tqdm==4.66.1 loguru==0.7.0 pydantic==1.10.8 fastapi==0.95.2 uvicorn==0.22.0")
    
    # Database
    print("\nInstalling database packages...")
    install_packages("sqlalchemy==2.0.20 pymongo==4.5.0 redis==4.6.0")
    
    # NLP
    print("\nInstalling NLP packages...")
    install_packages("textblob==0.17.1 vadersentiment==3.3.2")
    
    print("\nInstallation of essential packages complete.")
    print("Note: Some packages like ta-lib, tensorflow, and transformers may require special installation.")
    print("For ta-lib, follow: https://github.com/mrjbq7/ta-lib")
    print("For pytorch and tensorflow, consider using commands from their official websites.")

if __name__ == "__main__":
    main() 