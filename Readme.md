# Mulberry Disease Prediction - Complete Setup Guide
# =================================================

## 🌿 Project Overview
This project uses deep learning (CNN) to classify mulberry leaf diseases into 5 categories:
- Healthy Leaves
- Rust Leaves  
- Spot Leaves
- Deformed Leaves
- Yellow Leaves

The model achieves ~95% accuracy and includes both training scripts and visualization tools.

## 📁 Project Structure
```
Mulbary_disease_prediction/
├── Dataset/
│   └── Mulberry_Data/
│       ├── Healthy_Leaves/     (200 images)
│       ├── Rust_leaves/        (200 images)
│       ├── Spot_leaves/        (200 images)
│       ├── deformed_leaves/    (200 images)
│       └── Yellow_leaves/      (200 images)
├── Model/
│   └── mulberry_leaf_disease_model_enhanced.h5
├── train_model.py              (Training script)
├── Disease_pred.ipynb          (Visualization notebook)
├── Apps.py                     (Streamlit web app)
├── requirements_simple.txt     (Dependencies)
└── Readme.txt                  (This file)
```

## 🚀 Complete Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Windows 10/11 (instructions tested on Windows)
- At least 4GB RAM
- 2GB free disk space

### Step 1: Clone or Download Project
1. Clone the repository: `git clone <your-repo-url>`
2. Or download and extract the project zip file
3. Navigate to the project directory: `cd Mulbary_disease_prediction`
4. Open PowerShell in the project directory

### Step 2: Create Virtual Environment
```powershell
# Navigate to project directory
cd Mulbary_disease_prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
# Install all required packages
pip install -r requirements_simple.txt
```

**If you encounter errors, see Troubleshooting section below.**

### Step 4: Verify Installation
```powershell
# Test if all packages are installed correctly
python -c "import numpy, pandas, matplotlib, cv2, tensorflow, sklearn, seaborn; print('✅ All packages installed successfully!')"
```

## 🤖 Training the Model

### Option 1: Using Python Script (Recommended)
```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Run training script
python train_model.py
```

**Expected Results:**
- Training time: 30-60 minutes
- Final accuracy: ~95%
- Model saved to: `Model/mulberry_leaf_disease_model_enhanced.h5`
- Training plots saved as: `training_results.png`

### Option 2: Using Jupyter Notebook
```powershell
# Start Jupyter Notebook
jupyter notebook

# Open Disease_pred.ipynb and run cells in order
```

## 📊 Visualizing and Testing Results

### Using Jupyter Notebook
```powershell
# Start Jupyter
jupyter notebook

# Open Disease_pred.ipynb
# Run cells in order:
# 1. Import libraries
# 2. Configuration and setup  
# 3. Dataset analysis
# 4. Sample image display
# 5. Model loading and testing
# 6. Random sample testing
# 7. Comprehensive evaluation
# 8. Interactive prediction
```

### Using Streamlit Web App
```powershell
# Run the web application
streamlit run Apps.py
```

## 🔧 Troubleshooting

### Windows Long Path Support Error
**Error:** `[Errno 2] No such file or directory: ... tensorflow\include\external\...`

**Solution 1: Enable Long Path Support (Recommended)**
1. Open PowerShell as Administrator
2. Run: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
3. Restart computer
4. Try installation again

**Solution 2: Use Shorter Path**
1. Move project to shorter path: `C:\ML_Projects\mulberry_disease`
2. Create new virtual environment there
3. Install packages

**Solution 3: Install TensorFlow Separately**
```powershell
# Install other packages first
pip install numpy pandas matplotlib opencv-python pillow scikit-learn seaborn jupyter notebook h5py

# Install TensorFlow separately
pip install tensorflow --no-cache-dir --force-reinstall
```

### PowerShell Execution Policy Error
**Error:** `execution of scripts is disabled on this system`

**Solution:**
```powershell
# Set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Module Not Found Errors
**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
1. Ensure virtual environment is activated: `venv\Scripts\activate`
2. Install packages: `pip install -r requirements_simple.txt`
3. Verify installation: `python -c "import numpy; print('Success')"`

### Jupyter Notebook Not Using Virtual Environment
**Problem:** Jupyter uses system Python instead of virtual environment

**Solution:**
1. Install ipykernel in virtual environment: `pip install ipykernel`
2. Add virtual environment to Jupyter: `python -m ipykernel install --user --name=venv --display-name="Python (venv)"`
3. In Jupyter: Kernel → Change Kernel → Select "Python (venv)"

## 📋 Requirements File Contents
```
# Essential packages for Mulberry Disease Prediction
# Core ML and Deep Learning
tensorflow>=2.15.0
keras>=3.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Computer Vision and Image Processing
opencv-python>=4.8.0
Pillow>=10.0.0

# Data Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Jupyter Notebook Support
jupyter>=1.0.0
notebook>=6.5.0
ipykernel>=6.25.0

# Additional utilities
h5py>=3.9.0
```

## 🎯 Quick Start Commands
```powershell
# Complete setup in one go
python -m venv venv
venv\Scripts\activate
pip install -r requirements_simple.txt
python train_model.py
jupyter notebook
```

## 📈 Expected Performance
- **Model Accuracy:** ~95%
- **Training Time:** 30-60 minutes
- **Model Size:** ~1.6 GB
- **Inference Speed:** <1 second per image

## 🆘 Getting Help
If you encounter issues not covered here:
1. Check that all file paths are correct
2. Ensure virtual environment is activated
3. Verify all dependencies are installed
4. Try running individual components to isolate issues

## 📝 Notes
- The project works best with Python 3.11+
- GPU acceleration is optional but recommended for faster training
- All paths in the code are relative to the project root
- The model can be retrained with different parameters by modifying `train_model.py`

---
**Last Updated:** January 2025
**Tested on:** Windows 10/11, Python 3.11
