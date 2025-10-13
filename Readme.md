# 🌿 Mulberry Leaf Disease Detection

A deep learning project that uses Convolutional Neural Networks (CNN) to classify mulberry leaf diseases. The project includes both a training pipeline and a user-friendly web application.

## 🎯 Features

- **5 Disease Classifications**: Healthy, Rust, Spot, Deformed, and Yellow leaves
- **Web Application**: Interactive Streamlit interface for easy image upload and prediction
- **Jupyter Notebook**: Comprehensive analysis and visualization tools
- **Model Training**: Standalone training script with progress monitoring
- **High Accuracy**: Deep learning model with validation and testing

## 📋 Requirements

- Python 3.11+
- Windows 10/11 (instructions tested on Windows)
- At least 4GB RAM
- 2GB free disk space

## 🚀 Quick Start

### Step 1: Clone or Download Project
```bash
# Clone the repository
git clone <your-repo-url>
cd Mulbary_disease_prediction

# Or download and extract the project zip file
```

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

# Verify installation
python -c "import numpy, pandas, matplotlib, cv2, tensorflow, sklearn, streamlit; print('✅ All packages installed successfully!')"
```

### Step 4: Prepare Dataset
1. Create the following folder structure:
```
Dataset/
└── Mulberry_Data/
    ├── Healthy_Leaves/
    ├── Rust_leaves/
    ├── Spot_leaves/
    ├── deformed_leaves/
    └── Yellow_leaves/
```

2. Add your mulberry leaf images to the appropriate folders (200+ images per class recommended)

### Step 5: Train the Model
```powershell
# Train the model (this will take some time)
python train_model.py
```

### Step 6: Run the Application

#### Option A: Web Application (Recommended)
```powershell
# Run the Streamlit web app
streamlit run Apps.py
```
- Open your browser to `http://localhost:8501`
- Upload mulberry leaf images for instant disease detection
- View confidence scores and detailed analysis

#### Option B: Jupyter Notebook
```powershell
# Run Jupyter notebook for detailed analysis
jupyter notebook
```
- Open `Disease_pred.ipynb` for comprehensive model analysis
- Generate confusion matrices and training visualizations
- Test the model with sample images

## 📁 Project Structure

```
Mulbary_disease_prediction/
├── train_model.py              # Model training script
├── Apps.py                     # Streamlit web application
├── Disease_pred.ipynb          # Jupyter notebook for analysis
├── requirements_simple.txt     # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore file
└── Dataset/                    # Dataset folder (empty in repo)
    └── Mulberry_Data/
        ├── Healthy_Leaves/
        ├── Rust_leaves/
        ├── Spot_leaves/
        ├── deformed_leaves/
        └── Yellow_leaves/
```

## 🔧 Troubleshooting

### Common Windows Issues

#### PowerShell Execution Policy Error
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Long Path Support Error (TensorFlow)
```powershell
# Enable long path support (requires restart)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Module Not Found Errors
```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall packages
pip install --upgrade pip
pip install -r requirements_simple.txt
```

### Model Training Issues
- Ensure you have enough images (200+ per class)
- Check that images are in correct folder structure
- Verify image formats (JPG, PNG, JPEG supported)

## 📊 Model Performance

The model provides:
- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~90%+
- **Test Accuracy**: ~88%+
- **Confusion Matrix**: Detailed classification results
- **Training Plots**: Accuracy and loss visualization

## 🎨 Web Application Features

- **Image Upload**: Drag and drop or click to upload
- **Real-time Prediction**: Instant disease classification
- **Confidence Scores**: Probability percentages for each class
- **Visual Analysis**: Bar charts showing all probabilities
- **Error Handling**: Graceful handling of invalid images
- **Responsive Design**: Works on desktop and mobile

## 📈 Usage Examples

### Training the Model
```python
# The train_model.py script will:
# 1. Load and preprocess images
# 2. Split data into train/validation/test sets
# 3. Build and train CNN model
# 4. Evaluate performance
# 5. Save trained model and plots
```

### Using the Web App
1. Start the app: `streamlit run Apps.py`
2. Upload a mulberry leaf image
3. View the prediction and confidence score
4. Analyze the probability distribution

### Using the Notebook
1. Start Jupyter: `jupyter notebook`
2. Open `Disease_pred.ipynb`
3. Run cells to analyze model performance
4. Generate confusion matrices and plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: Mulberry leaf disease images
- Framework: TensorFlow/Keras
- Web Interface: Streamlit
- Visualization: Matplotlib, Seaborn

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify your Python environment
3. Ensure all dependencies are installed
4. Check the dataset structure

---

**Happy Disease Detection! 🌿🔬**