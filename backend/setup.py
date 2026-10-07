#!/usr/bin/env python3
"""
Setup script for RAG System
This script helps set up the environment and install dependencies
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 50)
    print(f"  {text}")
    print("=" * 50 + "\n")

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(e.stderr)
        return False

def check_python():
    """Check if Python is installed"""
    print_header("Checking Python Installation")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} found")
    return True

# Resolve project root path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
VENV_DIR = os.path.join(BACKEND_DIR, "venv")
REQUIREMENTS_PATH = os.path.join(BACKEND_DIR, "requirements.txt")

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print_header("Setting Up Virtual Environment")
    if os.path.exists(VENV_DIR):
        print("[INFO] Virtual environment already exists")
        return True
    
    escaped_venv_dir = VENV_DIR.replace('\\', '\\\\')
    if run_command(f"\"{sys.executable}\" -m venv \"{escaped_venv_dir}\"", "Creating virtual environment"):
        print("[OK] Virtual environment created successfully")
        return True
    return False

def get_pip_command():
    """Get the correct pip command based on OS"""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "pip")

def get_python_command():
    """Get the correct python command based on OS"""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def install_dependencies():
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    if not run_command(f"\"{pip_cmd}\" install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not os.path.exists(REQUIREMENTS_PATH):
        print(f"[ERROR] requirements.txt not found at {REQUIREMENTS_PATH}")
        return False
    
    if run_command(f"\"{pip_cmd}\" install -r \"{REQUIREMENTS_PATH}\"", "Installing requirements"):
        print("[OK] All dependencies installed successfully")
        return True
    return False

def download_nltk_data():
    """Download required NLTK data"""
    print_header("Downloading NLTK Data")
    python_cmd = get_python_command()
    
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("[OK] NLTK data downloaded")
except Exception as e:
    print(f"[WARNING] Could not download NLTK data: {e}")
    print("You may need to download it manually later")
"""
    
    result = subprocess.run([python_cmd, "-c", nltk_script], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    directories = [
        os.path.join(PROJECT_ROOT, "uploads"),
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(BACKEND_DIR, "dataset")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"[OK] Created directory: {os.path.basename(directory)}")
        else:
            print(f"[INFO] Directory already exists: {os.path.basename(directory)}")
    
    return True

def main():
    """Main setup function"""
    print_header("RAG System Setup")
    print("This script will set up your RAG System environment")
    print("It will:")
    print("  1. Check Python installation")
    print("  2. Create virtual environment")
    print("  3. Install dependencies")
    print("  4. Download NLTK data")
    print("  5. Create necessary directories")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    # Run setup steps
    steps = [
        ("Python Check", check_python),
        ("Virtual Environment", create_virtual_environment),
        ("Dependencies", install_dependencies),
        ("NLTK Data", download_nltk_data),
        ("Directories", create_directories),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n[ERROR] Setup failed at step: {step_name}")
            sys.exit(1)
    
    print_header("Setup Complete!")
    print("[OK] Your RAG System is ready to use!")
    print()
    print("To start the application:")
    if platform.system() == "Windows":
        print("  - Run: backend\\venv\\Scripts\\python backend\\app.py")
    else:
        print("  - Run: backend/venv/bin/python backend/app.py")
    print()
    print("The application will be available at: http://localhost:5000")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)

