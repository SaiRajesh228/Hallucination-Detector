#!/usr/bin/env python3
"""
Simple test to verify all components work
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_dependency_check():
    print("🧪 Running simple dependency check...")
    
    try:
        import sklearn
        print("✅ scikit-learn installed")
    except ImportError:
        print("❌ scikit-learn missing")
        return False
        
    try:
        import pandas as pd
        print("✅ pandas installed")
    except ImportError:
        print("❌ pandas missing")
        return False
        
    try:
        import numpy as np
        print("✅ numpy installed")
    except ImportError:
        print("❌ numpy missing")
        return False
        
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers installed")
    except ImportError:
        print("❌ sentence-transformers missing")
        return False
        
    try:
        import spacy
        print("✅ spacy installed")
    except ImportError:
        print("❌ spacy missing")
        return False
        
    print("✅ All dependencies are installed!")
    return True

def test_ollama():
    print("\n🔗 Testing Ollama connection...")
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is running")
            print("Available models:")
            print(result.stdout)
            return True
        else:
            print("❌ Ollama is not running properly")
            return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

if __name__ == "__main__":
    if simple_dependency_check():
        test_ollama()
        print("\n🎉 Everything is set up correctly!")
        print("You can now run: python main.py")
    else:
        print("\n❌ Please install missing dependencies first:")
        print("pip install scikit-learn pandas numpy sentence-transformers spacy")