#!/usr/bin/env python3
"""
Setup script for Hallucination Detection Framework
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("=" * 60)
    print("🛠️  Hallucination Detection Framework Setup")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ Please activate your virtual environment first!")
        print("\nTo create and activate virtual environment:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Mac/Linux)")
        print("  venv\\Scripts\\activate    # On Windows)")
        return
    
    print("✅ Virtual environment detected")
    
    # Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements. Please check your internet connection.")
        return
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  Warning: Could not download spaCy model. Using fallback sentence splitting.")
    
    # Verify Ollama is running and models are available
    print("\n🔍 Verifying Ollama setup...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if "llama3:latest" in result.stdout:
            print("✅ Ollama is running and llama3:latest model is available")
        else:
            print("⚠️  Warning: llama3:latest not found in available models")
            print("Available models:")
            print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama is not installed or not running")
        print("Please install Ollama from https://ollama.ai/")
        print("And pull the model: ollama pull llama3:latest")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the hallucination detection framework:")
    print("  python main.py")

if __name__ == "__main__":
    main()