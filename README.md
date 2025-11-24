# Brain Tumor Segmentation - Radiologist Application

A PyQt6-based desktop application for automated brain tumor segmentation using deep learning on multi-modal MRI scans.

## 🎯 Features

- **Multi-modal MRI Support**: Process FLAIR, T1, T1CE, and T2 sequences
- **Deep Learning Segmentation**: 3D U-Net architecture for accurate tumor detection
- **Interactive Visualization**: 2D slice viewer with adjustable overlay opacity
- **Animated Results**: Automatic GIF generation showing segmentation across slices
- **3D Visualization**: Optional Open3D-based 3D mesh rendering
- **Quantitative Metrics**: Dice scores, tumor volumes, and BraTS metrics
- **Export Capabilities**: Save results in NIfTI format and PNG panels

## 📋 Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, recommended for faster inference)
- 8GB+ RAM (16GB recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chbakar26/Brain-Tumor-Segmentation-From-MRI-Scans.git
cd brain-tumor-segmentation
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

```bash
# Copy the example configuration
cp config.example.py config_local.py

# Edit config_local.py with your model paths
```

Update `config_local.py` with your actual paths:
```python
MODEL_PATH = "path/to/your/best_brats_model_dice.pth"
MODEL_DEF_PATH = "path/to/your/improved3dunet.py"
```

## 📂 Model Setup

1. **Model Weights**: Place your trained `.pth` file in the project directory
2. **Model Definition**: Ensure `improved3dunet.py` is available
3. **Configuration**: Update paths in `config_local.py`

> **Note**: Model files are not included in this repository due to size constraints. You need to train or obtain a model separately.

## 💻 Usage

### Starting the Application

```bash
python brain_tumour_seg_app.py
```

### Workflow

1. **Load Data**: Click "📁 Load Patient Folder" and select a folder containing MRI scans
2. **Configure**: Adjust device (CPU/CUDA), base modality, and opacity settings
3. **Segment**: Click "🧠 SEGMENT" to run the segmentation
4. **View Results**: Explore 2D slices, animated GIF, and metrics
5. **3D Visualization**: (Optional) Click "📊 3D VISUALIZATION" for mesh rendering
6. **Export**: Results are automatically saved to the output directory

## 📁 Input Data Format

Patient folders should contain NIfTI files with the following naming:

```
patient_folder/
├── BraTS_001_flair.nii.gz
├── BraTS_001_t1.nii.gz
├── BraTS_001_t1ce.nii.gz
├── BraTS_001_t2.nii.gz
└── BraTS_001_seg.nii.gz (optional, for metrics)
```

Supported patterns:
- `*_flair.nii` or `*_flair.nii.gz`
- `*_t1.nii` or `*_t1.nii.gz`
- `*_t1ce.nii` or `*_t1ce.nii.gz`
- `*_t2.nii` or `*_t2.nii.gz`

## 📤 Output Structure

Results are saved to `seg_outputs/<patient_id>/run_<timestamp>/`:
```
seg_outputs/
└── BraTS_001/
    └── run_20240115-143022/
        ├── BraTS_001_pred.nii          # Segmentation mask
        ├── BraTS_001_overlay.gif       # Animated visualization
        └── BraTS_001_examples.png      # Example slices with legend
```

## 🖥️ Application Interface

![App Interface](app_interface/Screenshot%202025-11-23%20111553.jpg)


## 🎬 3D vizualization


https://github.com/user-attachments/assets/2a7e396d-b3d0-41d4-8081-ca209ef0834f

## 🌐 Live Demo
Try the web version of this application instantly on Hugging Face Spaces:
[**👉 Click here to open the Live App**](https://huggingface.co/spaces/abubakaroo7/Brain_tumour_segmentation)

<img width="494" height="415" alt="Screenshot 2025-11-24 180520" src="https://github.com/user-attachments/assets/750feb79-0406-459e-bfc1-b193b483104c" />


## 🎨 Segmentation Labels

| Label | Region | Color |
|-------|--------|-------|
| 0 | Background | Black |
| 1 | Necrotic Tumor Core | Red |
| 2 | Peritumoral Edema | Green |
| 3 | Enhancing Tumor | Yellow |


## 📊 Metrics Computed

- **Per-label Dice scores**: Accuracy for each tumor region
- **BraTS Composite Metrics**:
  - Whole Tumor (WT)
  - Tumor Core (TC)
  - Enhancing Tumor (ET)
- **Volume Measurements**: Tumor volumes in milliliters
- **Voxel Spacing**: Resolution information

## 🔧 Configuration Options

### Device Selection
- **auto**: Automatically use CUDA if available
- **cpu**: Force CPU processing
- **cuda**: Force GPU processing

### Inference Parameters
- **ZRANGE**: Z-axis crop range (e.g., "60:100")
- **PATCH_SIZE**: Sliding window size (e.g., "128,128,64")
- **OVERLAP**: Patch overlap ratio (0.0-1.0)
- **INTENSITY_NORM**: Normalization method ("zscore" or "minmax")

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `PATCH_SIZE` in config
- Enable `LOW_MEM_ACCUM = True`
- Use `FORCE_CPU = True`

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### 3D Visualization Not Working
```bash
pip install open3d scikit-image scipy
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BraTS Challenge organizers for dataset standards
- PyTorch and PyQt6 communities
- Open3D developers for 3D visualization tools

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{brain_tumor_seg_2024,
  title={Brain Tumor Segmentation Application},
  author={Abu bakar},
  year={2024},
  url={https://github.com/chbakar26/Brain-Tumor-Segmentation-From-MRI-Scans}
}
```
