# 🎨 AutoML Studio

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

AutoML Studio is a high-performance, intelligent end-to-end automated machine learning platform. It empowers anyone to upload tabular datasets, leverage "Dataset DNA" heuristics to automatically preprocess data, logically isolate the most appropriate algorithms, and train competitive ML models locally—all wrapped in a beautifully styled, dynamic user interface.

It features a twin-architecture: A **FastAPI** theoretical engine handling heavy computational pipelines and data logic, coordinated by a rich **Streamlit** user experience interface.

![AutoML Studio Dashboard](./assets/dashboard_preview.png)
*(Placeholder: Add a screenshot or GIF of the dashboard here)*

## ✨ Features

- **Dataset DNA Analyzer**: Instantly parses uploaded datasets (CSV, JSON, Excel, Parquet) to automatically determine shape, calculate missing value distributions, identify imbalances, and heuristically suggest target configurations.
- **Auto-Imputing & Auto-Encoding**: You never have to manually clean data again. The backend seamlessly applies `ColumnTransformers`, routing numeric data through Medians/StandardScalers and categorical data through Constant/OneHotEncoders safely.
- **Smart Model Selection (Pro Mode)**: It doesn't test blindly. It evaluates the exact shape and taxonomy of your dataset to dynamically build a tailored algorithmic roster (e.g. leveraging `SVM` for small datasets, and unleashing `XGBoost` for high-dimensional complexity).
- **Time Travel Training Logs**: View live metric updates as pipelines iteratively optimize.
- **Auto Report (Story Mode)**: Generates an automated "wrap-up" narrative explaining what data was analyzed, which algorithm dominated, and *why* it succeeded.
- **Deep Insights**: Explore exactly where the model fails via the Mistake Analyzer, view low-confidence classifications, and receive "Explain-Like-I'm-5" ML coaching strategies.
- **One-Click Deploy Bundles**: Automatically bundles and exports your trained `.pkl` model directly beside a custom-written `FastAPI` script, giving you a deployment-ready inference server in 1 click!

---

## 🏗️ Architecture

```text
AutoML Studio
├── frontend/
│   ├── app.py                # Main Streamlit config and entrypoint
│   ├── style.css             # Premium custom CSS properties (Glassmorphism/Google Fonts)
│   └── pages/
│       ├── 1_Home.py                    # Ingestion & Mode Configuration
│       ├── 2_DNA.py                     # Dataset Intelligence Profiling
│       ├── 3_Training_and_Results.py    # Live Orchestrator, Leaderboards, & Export
│       └── 4_Failure_Analyzer.py        # Edge Case testing and AI Insight Chat
├── backend/
│   ├── main.py               # FastAPI router handling multi-part ingestion & background tasks
│   └── core/
│       ├── data_profiler.py  # Dataset heuristic extraction logic
│       ├── model_trainer.py  # The Scikit-Learn/XGBoost dynamic training arena
│       ├── insights.py       # Narrative generation and AI Coaching synthesis
│       └── export.py         # ZIP creation containing model.pkl + dynamically generated API code
├── requirements.txt          # Shared dependencies
├── run.sh                    # Quick start execution script
└── README.md
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+ 
- `pip` package manager

### 1. Setup Environment
Clone the repository and set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Quick Start (Recommended)
You can launch both the FastAPI backend and Streamlit frontend concurrently using the provided shell script:
```bash
bash run.sh
```

### 4. Manual Launch
If you prefer to run them separately:
1. **Start the backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
2. **Start the frontend:** (in a new terminal, with the virtual environment activated)
```bash
streamlit run frontend/app.py
```

---

## 📖 How to Use

Once the application is live on `http://localhost:8501`, follow these steps:

1. **Upload Dataset**: Navigate to the **Home** tab and drag-and-drop your dataset (CSV, JSON, Excel, or Parquet).
2. **Review DNA**: Click on the **DNA** tab to review the automatic imputation plan and exploratory data analysis.
3. **Train Engine**: Go to **Training & Results** to start the parallel training pipeline. Watch the time-travel metrics update live.
4. **Export**: Once training completes, download the deployment-ready `.zip` bundle to serve your model immediately.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License.
# auto_ml
