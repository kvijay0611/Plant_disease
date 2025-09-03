# Plant Disease Detection

A machine learning project utilizing deep learning to detect and classify diseases in plants, aimed at supporting precision agriculture and crop protection efforts.

***

## 🚀 Live Demo
 
> [LIVE LINK](https://kvijay0611-plant-disease-app-fixed-e5hrhl.streamlit.app/)

***

## About the Project

This repository contains all code, experiments, and app deployment files for a plant disease detection system. The project leverages Convolutional Neural Networks (CNNs) to analyze plant images and identify disease categories. The system is intended to help farmers and agricultural professionals diagnose plant health issues quickly.

***

## Features

- **Image upload**: Detects diseases from plant leaf images using a trained CNN model.[1]
- **Multi-class prediction**: Classifies input images among several disease/health categories.
- **Web application**: Provides an intuitive interface for end users (built with Streamlit/Jupyter Notebook).

***

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kvijay0611/Plant_disease.git
   cd Plant_disease
   ```

2. **(Optional) Create & activate a virtual environment.**

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application:**
   ```bash
   streamlit run app_fixed.py
   ```
   
   or open the notebook for experimentation:
   ```bash
   jupyter notebook Plant_Disease_Detection.ipynb
   ```

***

## Project Structure

| Folder/File             | Purpose                                    |
|------------------------ |--------------------------------------------|
| main_app.py             | Streamlit web app interface                |
| Plant_Disease_Detection.ipynb | Jupyter Notebook for model training    |
| requirements.txt        | Python dependencies list                   |
| model/                  | Pre-trained model weights                  |
| data/                   | (Optional) Example datasets                |

***

## Usage

1. Launch the web app and open the provided address (e.g., `http://localhost:8501`).
2. Upload a plant leaf image.
3. The model predicts the disease/health class.

***

## Dataset

- Uses publicly available datasets such as PlantVillage or custom labeled data.
- [Add details on dataset sources, preprocessing, and licensing.]

***
