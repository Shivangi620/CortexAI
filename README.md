---
title: AutoML Studio
emoji: ✨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎨 AutoML Studio

This repository is configured to run on Hugging Face Spaces as a single Docker Space that serves:

- Streamlit frontend through Nginx on public port `7860`
- FastAPI backend internally on `127.0.0.1:8000`
- Celery worker plus Redis inside the same container for background training jobs

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green)

AutoML Studio is a high-performance, intelligent end-to-end automated machine learning platform. It empowers anyone to upload tabular datasets, leverage "Dataset DNA" heuristics to automatically preprocess data, logically isolate the most appropriate algorithms, and train competitive ML models locally through a standalone **HTML/CSS/JavaScript** frontend served directly by **FastAPI**.

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
│   └── web/
│       ├── index.html         # Standalone upload and training control surface
│       ├── results.html       # Redesigned Results Console
│       ├── styles.css         # Shared futuristic visual system
│       └── app.js             # Frontend logic for upload, train, inspect, predict, export
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
You can launch the FastAPI backend, worker stack, and built-in web frontend using the provided shell script:
```bash
bash run.sh
```

### 4. Manual Launch
If you prefer to run it manually:
1. **Start the backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
2. **Open the frontend:** `http://localhost:8000`

---

## 📖 How to Use

Once the application is live on `http://localhost:8000`, follow these steps:

1. **Upload Dataset**: Navigate to the **Home** tab and drag-and-drop your dataset (CSV, JSON, Excel, or Parquet).
2. **Review DNA**: Click on the **DNA** tab to review the automatic imputation plan and exploratory data analysis.
3. **Train Engine**: Go to **Training & Results** to start the parallel training pipeline. Watch the time-travel metrics update live.
4. **Export**: Once training completes, download the deployment-ready `.zip` bundle to serve your model immediately.

---

## 🌐 Deployment Options

### Quick Deploy (Recommended for Beginners)

#### 1. Railway (Easiest - 5 minutes)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init automl-studio
railway up
```

#### 2. Render (Free tier available)
```bash
# Run setup script
bash setup-render.sh

# Then deploy via Render dashboard
# Go to render.com and connect your GitHub repo
```

#### 3. Docker (Most Flexible)
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8000
```

### Advanced Deployment

#### AWS/GCP/Azure
- Use Elastic Beanstalk, App Engine, or App Service
- Add RDS/Cloud SQL for database persistence
- Use Cloud Storage for model artifacts

#### Heroku
```bash
# Create Heroku app
heroku create your-automl-studio

# Deploy
git push heroku main
```

### Production Considerations
- **Database**: Add PostgreSQL for production data persistence
- **Redis**: Required for background job queuing
- **Storage**: Use cloud storage (S3, GCS) for large model files
- **Scaling**: Consider load balancer for multiple instances
- **Security**: Add authentication, rate limiting, and input validation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License.
# auto_ml
# auto_ml
# auto_ml
