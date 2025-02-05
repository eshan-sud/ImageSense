
# Sets up the required python libraries/modules

# pip freeze > src/requirements.txt
# pip install scikit-image scikit-learn matplotlib tensorflow

# Install Python 3.13
# execute : pip install src/requirements.txt -y
# execute : venv\scripts\activate
# execure : streamlit run src/main.py

import os
import subprocess
import sys

def install_python():
    print("Downloading and installing Python 3.13...")
    python_installer_url = "https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe"
    installer_path = "python_installer.exe"
    subprocess.run(["curl", "-o", installer_path, python_installer_url], check=True)
    subprocess.run([installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1"], check=True)
    os.remove(installer_path)
    print("Python 3.13 installed successfully!")

def setup_virtualenv():
    print("Setting up virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("Virtual environment created successfully!")

def activate_virtualenv():
    print("Starting up the virtual environment...")
    activate_script = os.path.join("venv", "Scripts", "activate")
    subprocess.run([activate_script], shell=True)
    print("Virtual environment started successfully!")

def install_requirements():
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "src/requirements.txt"], check=True)
    print("Dependencies installed successfully!")

def activate_and_run():
    print("Activating virtual environment and running Streamlit app...")
    subprocess.run(["streamlit", "run", "src/main.py"], shell=True)

def main():
    install_python()
    setup_virtualenv()
    activate_virtualenv()
    install_requirements()
    activate_and_run()

if __name__ == "__main__":
    main()
