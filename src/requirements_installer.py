#!/usr/bin/env python3
"""
Requirements Installer for RBS Analysis Tool

This script checks for required packages and installs them if they are not present.
It ensures all dependencies are properly installed before running the main modules.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

# Required packages with their import names
REQUIRED_PACKAGES = {
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'networkx': 'networkx',
    'graphviz': 'graphviz',
    'folium': 'folium',
    'plotly': 'plotly',
    'dash': 'dash',
    'geopandas': 'geopandas',
    'shapely': 'shapely',
    'pandas': 'pandas',
    'numpy': 'numpy'
}

def check_package(package_name, import_name):
    """
    Check if a package is installed.
    
    Args:
        package_name (str): The name of the package in pip
        import_name (str): The name of the package when importing
        
    Returns:
        bool: True if installed, False otherwise
    """
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """
    Install a package using pip.
    
    Args:
        package_name (str): The name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except Exception as e:
        print(f"Error installing {package_name}: {e}")
        return False

def ensure_requirements():
    """
    Check all required packages and install missing ones.
    
    Returns:
        bool: True if all requirements are satisfied, False otherwise
    """
    all_installed = True
    missing_packages = []
    
    print("Checking required packages...")
    
    for pkg_name, import_name in REQUIRED_PACKAGES.items():
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_installed = False
    
    if all_installed:
        print("All required packages are already installed.")
        return True
    
    print(f"Missing packages: {', '.join(missing_packages)}")
    print("Attempting to install missing packages...")
    
    for pkg_name in missing_packages:
        success = install_package(pkg_name)
        if not success:
            all_installed = False
    
    if not all_installed:
        print("WARNING: Not all packages could be installed.")
        print("You may need to manually install the missing packages.")
        print("Use: pip install " + " ".join(missing_packages))
        return False
    
    print("All required packages have been installed successfully.")
    return True

def run_module(module_path):
    """
    Run a Python module after ensuring all requirements are installed.
    
    Args:
        module_path (str): Path to the module to run
        
    Returns:
        int: Return code from the executed module
    """
    if not ensure_requirements():
        print("ERROR: Required packages are missing. Cannot run the module.")
        return 1
    
    print(f"Running module: {module_path}")
    try:
        # Use subprocess to run the module
        result = subprocess.run([sys.executable, module_path])
        return result.returncode
    except Exception as e:
        print(f"Error running module {module_path}: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a module path is provided, run it after ensuring requirements
        module_path = sys.argv[1]
        sys.exit(run_module(module_path))
    else:
        # Otherwise just ensure requirements are installed
        if ensure_requirements():
            print("All requirements are satisfied. You can now run the RBS Analysis Tool.")
            sys.exit(0)
        else:
            print("Some requirements could not be installed.")
            sys.exit(1) 