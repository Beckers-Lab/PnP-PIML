# [ICRA 2025] Plug-and-Play Physics-Informed Machine Learning Using Uncertainty Quantified Port-Hamiltonian System

## Overview

This repository hosts the source code for the **PnP-PIML** framework—**Plug-and-Play Physics-Informed Machine Learning**—designed to enhance the prediction of obstacle dynamics in robotic systems. The framework integrates a physics-based modeling approach with modern machine learning techniques to deliver robust and accurate predictions, even under conditions that deviate from the training distribution. The “Plug-and-Play” aspect indicates that our method can be seamlessly incorporated into any existing prediction system to handle out-of-distribution scenarios effectively. Comprehensive details of the theoretical foundations and experimental validations are available in our accompanying publication.

The framework quantifies uncertainty from two primary perspectives:
1. **Pretrained LSTM Model Uncertainty:**  
   The uncertainty associated with predictions from the standard Long Short-Term Memory (LSTM) model is quantified using Conformal Prediction (CP). This provides a statistical guarantee on the model’s generalization performance under typical conditions.
2. **Out-of-Distribution Prediction Uncertainty:**  
   In scenarios where only limited observations are available, the uncertainty is estimated via a Bayesian Gaussian Process (GP) embedded within our physics-informed Port-Hamiltonian model. This approach facilitates a principled handling of uncertainties in novel or challenging dynamic environments.

## Project Structure

The repository is organized into several modules that correspond to the key components of the research:

- **Data Collection and Preprocessing:**  
  Contains computer vision tools and scripts for the acquisition and preprocessing of data.
  
- **Vanilla LSTM Model Implementation:**  
  Includes the baseline LSTM model augmented with Conformal Prediction for uncertainty quantification.
  
- **Physics-Informed Learning Module:**  
  Implements the proposed uncertainty quantified Port-Hamiltonian model using a Bayesian Gaussian Process for refining predictions in out-of-distribution scenarios.

## Getting Started

To set up the environment and run the code, please follow these instructions:

### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Beckers-Lab/PnP-PIML
```

### 2. Install Dependecies

```bash
conda create --name PnP_PIML python=3.9
conda activate PnP_PIML
pip install -r Requirements.txt
```

### 3. Running the Skeleton Algorithm

- **Setup:**  
  Navigate to the `./video/Data` directory and copy the `SgeNSke.py` script into your test data folder (e.g., `/video/Data/leftdown/`). Ensure that the corresponding video file (e.g., `leftdown.mp4`) is present.

- **Execution:**  
  Run the `SgeNSke.py` script. This will generate:
  - A video file named `output_downsampled_points.avi` that visualizes the processed output.
  - A MATLAB data file called `aligned_center_points.mat` containing the aligned center points.

- **Visualization:**  
  For quick visualization, execute the following MATLAB scripts sequentially:
  ```matlab
  video_start.m
  video_make.m

### 4. Running the Vanilla LSTM Model

- **Setup:**  
  Navigate to the `./Python_file` directory and copy the required `.mat` data file into this directory.

- **Execution:**  
  Execute the following command:
  ```bash
  python3 LSTM_baseline.py
  ```
    **This process will:**  
  - Train LSTM on dataset $D_0$
  - Calibrate Conformal Prediction using the large dataset
  - Test the model on limited new dynamic observations $D_1$
  - Retrain the model based on these limited observations.

### 5. Running the Physics-Informed Branch

- **Setup:**  
  Navigate to the `./Matlab_file` directory and ensure that the appropriate `.mat` data file is present in the current directory.

- **Execution:**  
  To perform the physics-informed prediction using the uncertainty quantified Port-Hamiltonian model, run:
  ```matlab
  gp_dphs_results.m
  ```
  Additionally, to conduct an ablation study using only the Gaussian Process (without the physics-informed Port-Hamiltonian structure), run:
  
   ```matlab
  Baseline_GP.m
  ```
  
## Citation

If you find this work useful in your research, please consider citing our publication:

> *[Kaiyuan Tan, Peilun Li, Jun Wang, Thomas Beckers]. (2025). Plug-and-Play Physics-Informed Machine Learning Using Uncertainty Quantified Port-Hamiltonian System. In Proceedings of ICRA 2025.*

## Acknowledgments

We gratefully acknowledge the support of Professor Beckers and Vanderbilt University. Special thanks to our collaborators and the broader research community for their insightful feedback and contributions.
