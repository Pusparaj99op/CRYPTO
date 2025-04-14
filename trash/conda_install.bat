@echo off
echo Installing dependencies for BNB Future AI Trading System...

:: Activate the conda environment
call conda activate richie_rich

:: Core scientific packages (from conda-forge)
echo Installing core scientific packages...
call conda install -c conda-forge numpy=1.24.3 pandas=2.0.3 scikit-learn=1.3.0 scipy=1.11.3 statsmodels matplotlib seaborn -y

:: Machine learning packages
echo Installing machine learning packages...
call conda install -c conda-forge tensorflow=2.13.0 -y
call conda install -c pytorch pytorch=2.0.1 -y
call conda install -c conda-forge xgboost=1.7.6 -y
call conda install -c conda-forge optuna=3.3.0 -y

:: Web packages
echo Installing web packages...
call conda install -c conda-forge flask=2.2.5 flask-sqlalchemy -y
call conda install -c conda-forge dash=2.13.0 plotly=5.16.1 -y

:: Database packages
echo Installing database packages...
call conda install -c conda-forge sqlalchemy=2.0.20 pymongo=4.5.0 redis-py=4.6.0 -y

:: Utilities
echo Installing utility packages...
call conda install -c conda-forge python-dotenv tqdm joblib loguru typing-extensions=4.5.0 -y

:: Install remaining packages with pip
echo Installing remaining packages with pip...
pip install python-binance==1.0.17 ccxt==3.1.54 pycoingecko==3.1.0
pip install ta==0.10.2 pandas-ta==0.3.14b0
pip install textblob==0.17.1 vaderSentiment==3.3.2
pip install flask-socketio==5.3.4 flask-restful==0.3.10
pip install pydantic==1.10.8 fastapi==0.95.2 uvicorn==0.22.0
pip install ollama==0.1.6

echo Installation complete!
echo You may need to manually install ta-lib using instructions from: https://github.com/mrjbq7/ta-lib 