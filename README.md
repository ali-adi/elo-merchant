# 🏆 Elo Merchant Category Recommendation Project

Welcome to the **Elo Merchant Category Recommendation** project! This repository contains everything you need to process data, train models, and evaluate performance effectively. 🚀

---

## 📂 Folder Structure

```
.
├── 📁 raw-data/                     # 📜 Original dataset files (not tracked by Git)
│   ├── 📄 Data_Dictionary.xlsx
│   ├── 📄 historical_transactions.csv
│   ├── 📄 merchants.csv
│   ├── 📄 new_merchant_transactions.csv
│   ├── 📄 sample_submission.csv
│   ├── 📄 test.csv
│   ├── 📄 train.csv
│
├── 📁 datasets/                      # 🔄 Processed datasets (not tracked by Git)
│   ├── 🗂️ final_dataset.parquet
│   ├── 🗂️ test.parquet
│   ├── 📂 split/
│   │   ├── 🗂️ train.parquet
│   │   ├── 🗂️ valid.parquet
│
├── 📁 elo-merchant/                  # 🔥 Project source code and notebooks
│   ├── 📖 README.md                  # 📌 Project documentation
│   ├── 📋 requirements.txt           # 🔧 Dependencies for the project
│   ├── 📁 env/                        # 🏗️ Virtual environment directory (not tracked by Git)
│   ├── 📂 1-data-handling/           # 📊 Data preparation and processing
│   │   ├── 📜 1-data-merging-train.ipynb
│   │   ├── 📜 2-data-merging-test.ipynb
│   │   ├── 📜 3-check-data.ipynb
│   │   ├── 📜 4-split-data.ipynb
│   │
│   ├── 📂 2-train/                   # 🎯 Training scripts and logs
│   │   ├── 📜 train-test.ipynb
│   │   ├── 📁 runs/                   # 📊 Training logs & outputs (not tracked by Git)
│   │   │   ├── 📂 [datetime]/         # 🕒 Each training run has a separate folder with current datetime
```

---

## 🚀 Quick Start Guide

### 📌 Setting Up the Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/elo-merchant-recommendation.git
   cd elo-merchant
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

1. 📂 **Ensure the dataset is placed in `raw-data/`**.
2. 🚀 **Run the notebooks in `1-data-handling/` sequentially:**
   - 🏗️ `1-data-merging-train.ipynb` → Merges training data.
   - 🔗 `2-data-merging-test.ipynb` → Merges test data.
   - 🔍 `3-check-data.ipynb` → Performs exploratory data analysis (EDA).
   - ✂️ `4-split-data.ipynb` → Splits the final dataset into train and validation sets.
3. 📁 Processed datasets will be saved in `datasets/`.

---

### 🏋️‍♂️ Training the Model

1. 📖 Open `2-train/train-test.ipynb`.
2. 🚀 Run the notebook to train and evaluate the model.
3. 📂 Training outputs will be stored in `2-train/runs/`, with a new folder for each run based on the **current datetime**.

---

### 🔍 Evaluating the Model

- 💾 **Model Checkpoints:** Automatically saved in `2-train/runs/[datetime]/model_checkpoint.pth`.
- 📊 **Training Logs & Plots:** Stored in `2-train/runs/[datetime]/`.
- 📜 **Test Predictions:** Saved in `2-train/runs/[datetime]/test_predictions.csv`.

---

### ✅ Best Practices

- 🔄 Use a **virtual environment** to manage dependencies.
- 🚫 Exclude large files from Git using `.gitignore`:
  ```
  env/
  2-train/runs/
  *.pyc
  __pycache__/
  ```

---

## 📜 License

📌 Add your license information here.

---

## 👏 Acknowledgements

🎉 Credit any **libraries, resources, or contributors** used in this project.

---

🚀 **Happy Coding!** 🎯


