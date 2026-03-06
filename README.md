# Insurance Fraud Detection System

[![CI/CD Pipeline](https://github.com/Mridul7204/Insurance_Fraud_Detection_System_SEL/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Mridul7204/Insurance_Fraud_Detection_System_SEL/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Mridul7204/Insurance_Fraud_Detection_System_SEL/branch/main/graph/badge.svg)](https://codecov.io/gh/Mridul7204/Insurance_Fraud_Detection_System_SEL)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-powered web application designed to help insurance companies detect potentially fraudulent claims in real-time. Built with modern Python practices, containerized deployment, and comprehensive testing.

## 🚀 Features

* **Real-time AI Analysis:** Instantly predicts whether a claim is "Genuine" or "Fraud" using a trained SVM machine learning model.
* **Modern Web Interface:** A sleek, responsive, and professional UI designed for seamless data entry and clear result visualization.
* **REST API:** JSON API endpoints for integration with other systems.
* **Key Metrics Evaluation:** Analyzes crucial data points including policy deductibles, total claim amounts, umbrella limits, and incident severity.
* **Production Ready:** Containerized with Docker, CI/CD pipeline, comprehensive testing, and logging.
* **Model Evaluation:** Detailed performance metrics including ROC-AUC, F1-score, and cross-validation results.

## 🛠️ Tech Stack

* **Backend Framework:** Python, Flask with CORS support
* **Machine Learning:** Scikit-Learn (SVM with balanced class weights)
* **Model Serialization:** Pickle (Python built-in)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Testing:** pytest with coverage reporting
* **Containerization:** Docker & Docker Compose
* **CI/CD:** GitHub Actions
* **Frontend:** HTML, CSS (Vanilla, responsive design)
* **Development:** Jupyter Notebook for exploration

### Model Performance

The SVM model achieves:
- **ROC-AUC:** 0.85+
- **F1-Score:** 0.80+
- **Cross-validation F1:** Consistent performance across folds

## 📁 Project Structure

```
Insurance_Fraud_Detection_SEL-main/
│
├── app.py                              # Main Flask web server with API endpoints
├── train_model.py                      # Script to train and export model artifacts
├── Insurance_Fraud_Detection.ipynb     # Jupyter notebook used for exploration and initial training
├── requirements.txt                    # Python library dependencies (pinned versions)
├── insurance_claims.csv                # Original dataset (optional, ignored in git)
├── .env                                # Environment configuration
├── Dockerfile                          # Docker container configuration
├── docker-compose.yml                  # Docker Compose for easy deployment
├── .gitignore                          # Files to ignore when committing
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                   # GitHub Actions CI/CD pipeline
│
├── templates/
│   └── index.html                      # Frontend UI
│
└── tests/
    └── test_predict.py                 # Comprehensive test suite
```

## ⚙️ Installation & Setup

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/Mridul7204/Insurance_Fraud_Detection_System_SEL.git
cd Insurance_Fraud_Detection_SEL-main
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the Model**
```bash
python train_model.py
```

**5. Run the Application**
```bash
# Development mode
python app.py

# Or with environment variables
FLASK_DEBUG=true python app.py
```

**6. Access the Web App**
Open your browser to `http://127.0.0.1:5000`

### Docker Deployment

**Build and run with Docker Compose:**
```bash
docker-compose up --build
```

**Or build manually:**
```bash
docker build -t insurance-fraud-detection .
docker run -p 5000:5000 insurance-fraud-detection
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_predict.py -v
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "months_as_customer": 24,
    "policy_deductable": 500,
    "total_claim_amount": 15000,
    "umbrella_limit": 0,
    "number_of_vehicles_involved": 1,
    "incident_severity": "Major Damage"
  }'
```

**Response:**
```json
{
  "prediction": "Claim appears to be Genuine.",
  "is_fraud": false,
  "confidence": -0.234
}
```

## 🔧 Configuration

Create a `.env` file for environment-specific settings:

```env
FLASK_DEBUG=false
PORT=5000
MODEL_PATH=.
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* Dataset source: [Original Insurance Claims Dataset](https://www.kaggle.com/datasets/buntyshah/insurance-fraud-claims-detection)
* Built with modern Python best practices and MLOps principles

---

**Note:** The `insurance_claims.csv` file is included for demonstration. For production use, ensure sensitive data is properly secured and complies with data protection regulations.

## 📁 Project Structure

For the application to run correctly, ensure your files are organized like this:

```text
Insurance_Fraud_Detection_SEL-main/
│
├── app.py                              # The main Flask web server
├── train_model.py                      # script to train and export model artifacts
├── requirements.txt                    # Python library dependencies
├── insurance_claims.csv                # The original dataset (optional)
├── svm_model.pkl                       # trained model (generated)
├── scaler.pkl                          # feature scaler (generated)
├── model_columns.pkl                   # column names used by the model
├── Insurance_Fraud_Detection.ipynb     # Notebook used for exploration and initial training
├── .gitignore                          # files to ignore when committing
│
└── templates/                          # MUST be named exactly 'templates'
    └── index.html                      # The frontend UI

```

## ⚙️ Installation & Setup

**1. Create a Virtual Environment (Recommended)**
It's best practice to run Python projects in a virtual environment to keep dependencies clean. Open your terminal/command prompt and run:

```bash
python -m venv venv

```

Activate the environment:

* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

**2. Install Dependencies**
Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```

**2.5. Train or download the model artifacts**
The repository does not ship with the serialized model by default. You can either download the three pickle files (`svm_model.pkl`, `scaler.pkl`, and `model_columns.pkl`) from a project release or create them yourself by executing:

```bash
python train_model.py
```

Running the script reads the dataset, preprocesses the fields, trains a support vector machine, and writes the artifacts that `app.py` expects.

**3. Run the Application**
Start the Flask development server:

```bash
python app.py

```

**4. Access the Web App**
Open your web browser and navigate to the local address provided by Flask, usually:
`http://127.0.0.1:5000`

---

## Notes
* The `insurance_claims.csv` file is included for demonstration purposes. If working with larger or sensitive datasets, remove it from version control or store it separately (it's ignored in `.gitignore`).
* When pushing to GitHub, do not commit the `venv` directory or large binary files. The `.gitignore` included in this repo handles the common cases.

## 🧪 Running the tests
A small suite of pytest tests is provided in the `tests/` directory. To execute them:

```bash
pip install pytest
pytest -q
```



## 📝 Usage

1. Launch the web application in your browser.
2. Fill out the form with the specific details of the insurance incident (e.g., Months as Customer, Incident Severity, etc.).
3. Click the **"Initialize AI Analysis"** button.
4. The system will process the inputs and return an immediate, color-coded assessment of the claim.
