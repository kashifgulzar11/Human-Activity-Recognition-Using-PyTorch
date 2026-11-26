# Human Activity Recognition Using PyTorch
### Custom BiLSTM & Temporal Attention on Smartphone Sensor Data

**Course:** Applied Machine Learning  
**Semester:** Fall 2026  
**Assignment:** 04 — Human Activity Recognition from Phone Motion Sensors

---

## 1. Project Overview

This project focuses on Human Activity Recognition (HAR) using smartphone motion sensor data and a custom deep learning architecture implemented in PyTorch.

The objective is to classify six human physical activities using raw inertial sensor signals collected from smartphones.

We developed a custom architecture called **HARNet**, which combines a Bidirectional Long Short-Term Memory (BiLSTM) network with a Temporal Attention mechanism to learn important patterns from sequential sensor data.

The model processes nine sensor channels over 128 time steps and predicts one of six activities.

## 2. Dataset

We used the **UCI Human Activity Recognition Using Smartphones Dataset**.

**Dataset Source:**  
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset contains smartphone sensor recordings collected from 30 volunteers performing six different activities.

### Dataset Details

| Description | Value |
|---|---:|
| Training samples | 7,352 |
| Testing samples | 2,947 |
| Validation volunteers | 3, 15, 25, 30 |
| Sensor channels | 9 |
| Time steps per sample | 128 |
| Number of classes | 6 |
| Input shape | (Batch Size, 128, 9) |

### Activity Classes

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

### Sensor Signals

The model uses nine raw inertial sensor channels:

- Body Acceleration X
- Body Acceleration Y
- Body Acceleration Z
- Body Gyroscope X
- Body Gyroscope Y
- Body Gyroscope Z
- Total Acceleration X
- Total Acceleration Y
- Total Acceleration Z

**Important:** This project uses the nine raw 128-step inertial signal channels, not the 561 engineered features.

## 3. Project Methodology

The project follows these main stages:

1. Dataset loading and exploration.
2. Sensor signal visualization.
3. Training and validation split.
4. Data standardization.
5. Custom recurrent neural network implementation.
6. BiLSTM and Temporal Attention architecture.
7. Model training and validation.
8. Model evaluation.
9. Attention visualization.
10. Streamlit application development.

### Data Preparation

- Loaded the official UCI HAR training and testing partitions.
- Used volunteers `[3, 15, 25, 30]` from the official training partition for validation.
- Kept the official test volunteers separate.
- Converted activity labels from `1–6` to `0–5`.
- Calculated the mean and standard deviation using only the training partition.
- Applied the same training statistics to validation and test data.

### Model Architecture

The proposed model is called **HARNet**.

Its architecture is:

```text
Input Sensor Data
(B, 128, 9)
       |
       v
Custom BiLSTM
(B, 128, 128)
       |
       v
Temporal Attention
       |
       v
Context Vector
(B, 128)
       |
       v
Linear Classifier
       |
       v
Output
(B, 6)
```

### Custom Implementations

The following components were implemented manually using PyTorch:

- Vanilla RNN Cell
- LSTM Cell
- LSTM Sequence Layer
- Bidirectional LSTM (BiLSTM)
- Temporal Attention Mechanism
- HARNet Classification Model

Built-in recurrent layers and attention modules prohibited by the assignment were not used.

### Architecture Configuration

| Parameter | Value |
|---|---:|
| Input dimensions (D) | 9 |
| Hidden dimensions (H) | 64 |
| Sequence length (T) | 128 |
| Number of BiLSTM directions | 2 |
| BiLSTM output dimensions | 128 |
| Number of classes | 6 |
| Trainable parameters | 46,982 |

## 4. Model Training

The model was trained using the following configuration:

| Training Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 30 |
| Loss Function | Cross Entropy Loss |
| Gradient Clipping | 1.0 |
| Checkpoint Selection | Lowest Validation Loss |

The best model checkpoint was selected using validation loss rather than test performance.

**Saved Model:**

```text
results/best_model.pt
```

## 5. Results and Performance

The completed model achieved the following test performance:

<details>
<summary><strong>Model Performance</strong></summary>

| Metric | Result |
|---|---:|
| Test Accuracy | 89.99% |
| Best Validation Loss Epoch | 19 |
| Trainable Parameters | 46,982 |
| Official Test Samples | 2,947 |

</details>

### Confusion Matrix

The confusion matrix is used to evaluate the classification performance across all six activity classes.

The diagonal values represent correctly classified samples, while the off-diagonal values represent classification errors.

The confusion matrix contains all 2,947 official test samples.

### Attention Visualization

Temporal Attention helps the model identify the most important time steps for activity classification.

The project generates:

- Six individual attention visualizations.
- Six average attention curves.
- Mean maximum-attention values for each activity class.
- Sensor signal and attention plots.

These visualizations help understand how the model focuses on different parts of the input sequence.

## 6. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Pandas | Data Handling |
| Streamlit | Interactive Web Application |
| Git | Version Control |
| GitHub | Project Repository |

## 7. Streamlit Application

An interactive Streamlit application was developed to demonstrate the Human Activity Recognition model.

The application provides a user-friendly interface for interacting with the trained model.

### Application Features

- Interactive Human Activity Recognition interface.
- Sensor signal visualization.
- Activity prediction.
- Six-class prediction probabilities.
- Temporal attention visualization.

The application can be launched using:

```bash
streamlit run app/app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

## 8. Project Structure

```text
Human-Activity-Recognition-PyTorch/
│
├── app/
│   └── app.py
│
├── blog/
│   ├── medium_blog_outline.md
│   └── linkedin_post_template.md
│
├── data/
│   └── UCI HAR Dataset/
│
├── results/
│   ├── example_windows.png
│   ├── best_model.pt
│   ├── analysis.md
│   ├── report_tables_template.md
│   └── README.md
│
├── scripts/
│   ├── check_environment.py
│   └── extract_nested_zips.py
│
├── src/
│   ├── load_data.py
│   ├── plot_windows.py
│   ├── prepare.py
│   ├── layers.py
│   ├── model.py
│   ├── train.py
│   ├── metrics.py
│   └── test_model.py
│
├── tests/
│   └── test_shapes.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Human-Activity-Recognition-PyTorch
```

### Step 2: Create a Virtual Environment

For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the UCI HAR dataset from:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract the dataset under the project's `data/` directory.

If the downloaded archive contains another ZIP file, extract it using:

```bash
python scripts/extract_nested_zips.py
```

## 10. How to Run the Project

Run the following commands from the project root directory.

### Step 1: Check Environment

```bash
python scripts/check_environment.py
```

### Step 2: Load Dataset

```bash
python -m src.load_data
```

### Step 3: Visualize Sensor Signals

```bash
python -m src.plot_windows
```

Generated visualization:

```text
results/example_windows.png
```

### Step 4: Prepare Dataset

```bash
python -m src.prepare
```

### Step 5: Verify Model Architecture

```bash
python -m src.model
```

### Step 6: Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

### Step 7: Train the Model

```bash
python -m src.train
```

### Step 8: Evaluate the Model

```bash
python -m src.test_model
```

### Step 9: Launch Streamlit Application

```bash
streamlit run app/app.py
```

**Note:** Training may take longer when running on a CPU.

## 11. Important Implementation Rules

The implementation follows the assignment's critical requirements:

- No `torch.nn.RNN`, `torch.nn.LSTM`, or `torch.nn.GRU`.
- No built-in recurrent cell variants.
- No `torch.nn.MultiheadAttention`.
- No scaled-dot-product attention.
- No scikit-learn, TensorFlow, or Keras.
- Validation volunteers are exactly `[3, 15, 25, 30]`.
- Standardization statistics are calculated using training data only.
- Official test data is kept separate during training and validation.
- The best checkpoint is selected using validation loss.
- Gradient clipping is applied after backpropagation and before optimizer updates.

## 12. Learning Outcomes

Through this project, we gained practical experience in:

- Understanding raw smartphone motion sensor data.
- Time-series data preprocessing and standardization.
- Implementing custom recurrent neural network components.
- Understanding LSTM and BiLSTM architectures.
- Implementing temporal attention mechanisms.
- Training and evaluating deep learning models.
- Creating confusion matrices and per-class accuracy metrics.
- Visualizing attention weights.
- Developing interactive machine learning applications using Streamlit.
- Managing collaborative projects using Git and GitHub.

## 13. Team Members

**Project Contributors:**

- **Muhammad Umar Draz**
- **Kashif Gulzar**

Both team members contributed to the project development and implementation.

---

## 14. References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2013.

2. UCI Machine Learning Repository — Human Activity Recognition Using Smartphones Dataset:  
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

3. PyTorch Documentation:  
   https://pytorch.org/docs/stable/index.html

4. Streamlit Documentation:  
   https://docs.streamlit.io/

---

**Developed as part of Applied Machine Learning — Assignment 04, Fall 2026.**
