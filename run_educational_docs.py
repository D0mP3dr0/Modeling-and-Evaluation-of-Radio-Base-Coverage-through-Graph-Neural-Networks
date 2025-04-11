#!/usr/bin/env python3
"""
RBS Analysis - Educational Documentation Runner

This script runs the educational documentation module, ensuring that
all required packages are installed first.
"""

import sys
import os
from pathlib import Path

def main():
    """Run the educational documentation module"""
    
    # Get the absolute path to the module
    script_dir = Path(__file__).resolve().parent
    installer_path = script_dir / "src" / "requirements_installer.py"
    educational_docs_path = script_dir / "src" / "educational_documentation.py"
    
    # Check that the paths exist
    if not installer_path.exists():
        print(f"ERROR: Cannot find requirements installer at {installer_path}")
        return 1
        
    if not educational_docs_path.exists():
        print(f"ERROR: Cannot find educational documentation module at {educational_docs_path}")
        return 1
    
    print("=" * 80)
    print("RBS Analysis - Running Educational Documentation Module")
    print("=" * 80)
    print(f"Using Python: {sys.executable}")
    print(f"Module path: {educational_docs_path}")
    print("=" * 80)
    
    # Import and run the requirements installer
    # This will ensure all required packages are installed
    try:
        # Add the src directory to the path
        if script_dir not in sys.path:
            sys.path.append(str(script_dir))
        
        # Import the installer module
        sys.path.append(str(script_dir / "src"))
        from src.requirements_installer import run_module
        
        # Run the educational documentation module
        return run_module(str(educational_docs_path))
    
    except ImportError:
        # If we can't import the installer, try running it directly
        print("Running requirements installer as a subprocess...")
        import subprocess
        
        # First run the installer to ensure requirements
        subprocess.check_call([sys.executable, str(installer_path)])
        
        # Then run the educational documentation module
        return subprocess.check_call([sys.executable, str(educational_docs_path)])

if __name__ == "__main__":
    sys.exit(main()) 