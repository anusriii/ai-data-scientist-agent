# 🧪 AI Data Scientist Agent

An end-to-end AI Data Scientist Agent that automates data analysis, exploratory data analysis, machine learning, deep learning, model evaluation, explainability, and report generation.

The goal is simple:

> Upload a dataset → Select the target → Run the AI Data Scientist.

The system automatically handles the technical workflow.

---

## 🚀 Features

### 📂 Data Ingestion

- CSV support
- Excel support
- Automatic dataset loading
- Dataset preview

### 📊 Automated Data Profiling

- Dataset dimensions
- Missing-value analysis
- Duplicate detection
- Numerical columns
- Categorical columns
- Unique-value analysis
- Statistical summaries

### 📈 Exploratory Data Analysis

- Numerical distributions
- Histograms
- Correlation analysis
- Dataset statistics
- Interactive Streamlit visualizations

### 🧹 Data Cleaning

The system can automatically handle:

- Missing values
- Duplicate rows
- Numerical preprocessing
- Categorical preprocessing
- Feature preparation

### 🤖 Automated Machine Learning

The agent automatically:

- Detects the problem type
- Selects appropriate models
- Preprocesses features
- Trains multiple models
- Evaluates models
- Compares model performance
- Identifies the best-performing model

Supported model families include:

- Logistic Regression
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting

### 🧠 Deep Learning

The project also includes a deep-learning pipeline for more advanced modeling.

The goal is to allow the agent to compare traditional machine learning with neural-network approaches.

### 🧠 Explainability

The system provides model explainability through feature-importance analysis.

This helps answer:

> Which features contributed most to the model?

### 📄 Automated Reports

The agent can generate an HTML report containing analysis results.

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │       User          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Ingestion      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Profiling      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Cleaning       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ Machine Learning│       │ Deep Learning   │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Model Evaluation    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Explainability      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Report Generation   │
                         └─────────────────────┘