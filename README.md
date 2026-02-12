# Intracranial Aneurysm Detection using 3D Deep Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Automated detection of intracranial aneurysms in brain CT scans using 3D convolutional neural networks. Built for the RSNA 2024 Intracranial Aneurysm Detection Challenge.

##  Project Overview

Intracranial aneurysms are potentially life-threatening bulges in brain blood vessels. Early detection is critical but challenging due to:
- Small size of aneurysms (often <5mm)
- Complex 3D vascular anatomy
- Variability across 13 different anatomical locations

This project implements a **3D U-Net architecture** to automatically detect and localize aneurysms in volumetric CT scans, achieving approximately **50% detection accuracy** on the test set.

##  Why This Matters

Manual review of brain CT scans is time-intensive and prone to human error. Automated detection systems can:
- Flag suspicious cases for radiologist review
- Reduce missed diagnoses
- Prioritize urgent cases
- Support radiologists in high-volume settings

##  Technical Approach

### Architecture: 3D U-Net
- **Why U-Net?** Gold standard for medical image segmentation with encoder-decoder structure and skip connections
- **Why 3D?** Aneurysms are volumetric structures requiring spatial context across all three dimensions
- **Model specs**: ~2.5M parameters, processes 64×128×128 volumes

### Key Components
1. **Data Processing**: DICOM medical image format handling, Hounsfield unit normalization
2. **Deep Learning**: PyTorch-based 3D convolutional neural network
3. **Multi-label Classification**: Predicts 14 outputs (13 anatomical locations + overall presence)
4. **Memory Optimization**: Techniques to handle large 3D volumes on limited GPU memory

##  Results

- **Competition Score**: ~50% accuracy
- **Dataset**: RSNA 2024 competition data (brain CT scans with expert annotations)
- **Challenge**: Severe class imbalance (most scans negative for aneurysms)

### What Worked
 3D convolutions captured spatial context  
 U-Net architecture preserved fine details  
 Data normalization improved training stability  

### Areas for Improvement
 Larger model capacity  
 Advanced data augmentation  
 Ensemble methods  
 Attention mechanisms  
 Alternative loss functions (Dice, Focal)  

##  Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA-capable GPU (recommended)
```

### Installation
```bash
git clone https://github.com/yourusername/aneurysm-detection-rsna.git
cd aneurysm-detection-rsna
pip install -r requirements.txt
```

### Required Libraries
- `torch` - Deep learning framework
- `pydicom` - DICOM medical image reading
- `nibabel` - NIfTI segmentation mask handling
- `numpy` - Numerical operations
- `scikit-image` - Image preprocessing
- `pandas` - Data manipulation
- `tqdm` - Progress bars

### Usage

See the [main notebook](aneurysm_detection_cleaned.ipynb) for full implementation details.

Basic inference example:
```python
from model import UNet3D, predict_single_scan

# Load trained model
model = UNet3D(in_channels=1, num_classes=14)
model.load_state_dict(torch.load('model_weights.pth'))

# Predict on new scan
predictions = predict_single_scan(model, 'path/to/dicom/series', device='cuda')
# Returns 14 probability scores
```

##  Project Structure

```
aneurysm-detection-rsna/
├── aneurysm_detection_cleaned.ipynb  # Main notebook with full pipeline
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── results/                          # Sample predictions and visualizations
└── docs/                            # Additional documentation
```

##  Medical Imaging Concepts

### DICOM Format
Standard format for medical images. Each CT scan consists of 100-300 2D slices that form a 3D volume.

### Anatomical Locations
The model predicts aneurysm presence in 13 specific brain arteries:
- Internal Carotid Arteries (left/right, infraclinoid/supraclinoid)
- Middle Cerebral Arteries (left/right)
- Anterior Cerebral Arteries (left/right)
- Posterior Communicating Arteries (left/right)
- Anterior Communicating Artery
- Basilar Tip
- Other Posterior Circulation

### Hounsfield Units
CT scans measure tissue density in Hounsfield Units (HU):
- Air: -1000 HU
- Water: 0 HU
- Bone: +1000 HU
Values are normalized for neural network input.

##  What I Learned

**Technical Skills:**
- Implementing 3D convolutional neural networks from scratch
- Working with medical imaging standards (DICOM, NIfTI)
- Managing GPU memory with large volumetric data
- Multi-label classification for imbalanced datasets

**Domain Knowledge:**
- Brain vascular anatomy
- Clinical requirements for medical AI (interpretability, accuracy thresholds)
- Challenges in medical ML (limited data, regulatory constraints)

##  Future Work

1. **Error Analysis**: Systematic review of false positives/negatives
2. **Model Improvements**: Attention mechanisms, multi-scale processing
3. **Visualization**: GradCAM for 3D to understand model focus areas
4. **Clinical Validation**: Testing with radiologists for real-world applicability

##  References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (2016)
- RSNA 2024 Intracranial Aneurysm Detection Challenge

##  License

MIT License - feel free to use this code for learning or research purposes.

##  Acknowledgments

- RSNA for hosting the competition and providing the dataset
- Kaggle community for discussions and shared approaches
- Medical imaging researchers whose papers informed this work

##  Contact

Questions or suggestions? Feel free to open an issue or reach out!

---

**Note**: This is a learning/research project. The model is not validated for clinical use and should not be used for medical diagnosis.
