# Elo Merchant Category Recommendation Project

This repository contains the code and data for the **Elo Merchant Category Recommendation** project. The project is divided into two main parts: **data handling** and **model training**. Data handling is performed using Jupyter notebooks to merge, clean, and split the data, while model training is done using a separate notebook that includes **checkpointing, logging, and evaluation**.

## 📂 Folder Structure

The project is structured as follows:

```
.
├── 1-data-handling
│   ├── 1-data-merging-train.ipynb    # Merges training data with additional sources
│   ├── 2-data-merging-test.ipynb     # Merges test data with additional sources
│   ├── 3-check-data.ipynb            # Performs exploratory data analysis (EDA) and data quality checks
│   └── 4-split-data.ipynb            # Splits the final dataset into train/validation sets
├── 2-train
│   ├── runs                          # Folder where training outputs (checkpoints, logs, plots, predictions) are saved.
│   │   └── [datetime folders]        # Each training run creates a new folder (named with the current datetime)
│   └── train-test.ipynb              # Notebook for model training and evaluation
└── env                               # Virtual environment folder (not tracked in Git)

# External Datasets (Not included in repository)
../datasets
│   ├── final_dataset.parquet         # Final aggregated dataset after merging all sources
│   ├── test.parquet                  # Test dataset for prediction (without target)
│   └── split
│       ├── train.parquet             # Training split of the final dataset (with target)
│       └── valid.parquet             # Validation split of the final dataset (with target)
```

---

## 🚀 Quick Start Guide

### 📌 Setting Up the Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/elo-merchant-recommendation.git
   cd elo-merchant-recommendation
   ```

2. **Create and activate a virtual environment:**
   - macOS/Linux:
     ```bash
     python -m venv env
     source env/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv env
     env\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### 📊 Data Handling Workflow

- **Step 1:** Ensure the dataset is placed in the correct external directory (`../datasets`).
- **Step 2:** Run the notebooks in `1-data-handling`:
  - `1-data-merging-train.ipynb`: Merges training data
  - `2-data-merging-test.ipynb`: Merges test data
  - `3-check-data.ipynb`: Performs exploratory data analysis (EDA)
  - `4-split-data.ipynb`: Splits the final dataset into train and validation sets
  
- **Step 3:** The cleaned datasets will be stored in `../datasets/split/`

---

### 🏋️‍♂️ Training the Model

- **Step 1:** Open `2-train/train-test.ipynb`
- **Step 2:** Run the notebook to train and evaluate the model
- **Step 3:** Training outputs will be saved automatically in `2-train/runs/` with a unique datetime folder

---

### 🔍 Evaluating the Model

- **Saved Model Checkpoints:** Model weights are saved automatically in `runs/[datetime]/model_checkpoint.pth`
- **Training Logs & Plots:** Loss logs and plots are saved in `runs/[datetime]/`
- **Test Predictions:** After training, test predictions can be generated and saved in `runs/[datetime]/test_predictions.csv`

---

### ✅ Best Practices

- **Use a virtual environment** to keep dependencies clean.
- **Exclude large files** from Git by adding them to `.gitignore`:
  ```
  env/
  2-train/runs/
  *.pyc
  __pycache__/
  ```
- **Document your changes** using clear commit messages.
- **Use Git LFS** for large datasets if needed:
  ```bash
  git lfs install
  git lfs track "*.parquet"
  ```

---

## 📜 License

Add your license information here.

---

## 👏 Acknowledgements

Credit any **libraries, resources, or contributors** used in this project.

---

🚀 Happy Coding! 🎯


