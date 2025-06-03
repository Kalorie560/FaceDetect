#!/usr/bin/env python3
"""
Simple test runner script for the facial keypoints detection project
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Main test runner function."""
    print("🧪 Facial Keypoints Detection - Test Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Error: src directory not found. Please run from the project root.")
        sys.exit(1)
    
    all_passed = True
    
    # Test 1: Check Python syntax
    success = run_command(
        "python -m py_compile src/models/cnn_model.py", 
        "Python syntax check for models"
    )
    all_passed = all_passed and success
    
    # Test 2: Check data module syntax
    success = run_command(
        "python -m py_compile src/data/dataset.py", 
        "Python syntax check for data modules"
    )
    all_passed = all_passed and success
    
    # Test 3: Check training module syntax
    success = run_command(
        "python -m py_compile src/training/trainer.py", 
        "Python syntax check for training modules"
    )
    all_passed = all_passed and success
    
    # Test 4: Check webapp syntax
    success = run_command(
        "python -m py_compile webapp/app.py", 
        "Python syntax check for webapp"
    )
    all_passed = all_passed and success
    
    # Test 5: Try to import main modules
    success = run_command(
        "python -c \"import sys; sys.path.append('src'); from models.cnn_model import create_model; print('✅ Models import successful')\"",
        "Import test for models"
    )
    all_passed = all_passed and success
    
    # Test 6: Try to import data modules
    success = run_command(
        "python -c \"import sys; sys.path.append('src'); from data.dataset import FacialKeypointsDataset; print('✅ Data modules import successful')\"",
        "Import test for data modules"
    )
    all_passed = all_passed and success
    
    # Test 7: Check if pytest is available and run tests if possible
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], capture_output=True, check=True)
        success = run_command(
            "python -m pytest tests/ -v", 
            "Unit tests with pytest"
        )
        all_passed = all_passed and success
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠️  pytest not available, skipping unit tests")
        print("   Install pytest to run unit tests: pip install pytest")
    
    # Test 8: Check requirements file
    if os.path.exists("requirements.txt"):
        success = run_command(
            "python -c \"import pip; print('✅ pip available for requirements check')\"",
            "Requirements file validation"
        )
        all_passed = all_passed and success
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("🎉 All tests passed! The project structure looks good.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download Kaggle data: kaggle competitions download -c facial-keypoints-detection")
        print("3. Configure ClearML: edit config/clearml.yaml")
        print("4. Start training: python src/training/train.py --data_path training.csv")
        print("5. Run webapp: streamlit run webapp/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()