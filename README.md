# 🚢 Titanic Survival Prediction

This repository contains a **Logistic Regression** model to predict the survival of Titanic passengers, along with a **Streamlit web application** for interactive predictions.

---

## 🧠 Project Overview

This project is based on the classic **Titanic survival prediction** problem. Using **logistic regression**, we aim to predict whether a passenger survived based on their features such as age, sex, class, etc. The pipeline includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing
- 🔍 Model Building & Evaluation
- 📈 Model Interpretation
- 🌐 Deployment using **Streamlit**

---

## 📁 Repository Structure

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `Logistic Regression.ipynb` | Complete ML workflow including EDA, preprocessing, training, evaluation     |
| `Logistic Regression.docx`  | Project overview + interview questions on logistic regression               |
| `titanic_app.py`           | Streamlit web app script for making survival predictions                   |
| `Titanic_train.csv`        | Training dataset for model development                                     |
| `Titanic_test.csv`         | Test dataset for model evaluation and submission                          |
| `gender_submission.csv`    | Sample submission format (used in Kaggle competitions)                    |
| `titanic_model.pkl`        | Serialized logistic regression model (created after training notebook run) |

---

## 🔄 Workflow

### 1️⃣ Data Exploration

- Analyze data distributions (e.g., age, sex, class)
- Visualize survival trends across features using Seaborn/Matplotlib

### 2️⃣ Data Preprocessing

- Handle missing values (e.g., age, embarked)
- Encode categorical variables (`Sex`, `Embarked`, `Pclass`)
- Create new features if needed (e.g., family size)

### 3️⃣ Model Building

- Train a **Logistic Regression** model using `scikit-learn`
- Split data into train/test sets
- Save model as `titanic_model.pkl` using `joblib`

### 4️⃣ Model Evaluation

Evaluate model using metrics such as:
- ✅ **Accuracy**
- 📊 **Precision**
- 📉 **Recall**
- ⚖️ **F1 Score**
- 📈 **ROC-AUC Score**

### 5️⃣ Deployment with Streamlit

- User inputs values like age, gender, ticket class
- Model returns survival prediction (0 = Did not survive, 1 = Survived)

---

## ▶️ How to Run

### 1. Create and Activate Virtual Environment (Optional but Recommended)
Bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

2. Install Required Libraries
pip install pandas scikit-learn matplotlib seaborn joblib streamlit

4. Train the Model
Run the Jupyter notebook to:

Load and preprocess data

Train the logistic regression model

Save the model as titanic_model.pkl

Bash 

jupyter notebook Logistic\ Regression.ipynb

4. Run the Streamlit App
Once the model is saved:

streamlit run titanic_app.py



## 🧠 Interview Questions
1. What is the difference between Precision and Recall?
Metric	Description
Precision	Measures the accuracy of positive predictions. It answers: “Of all the predicted positive cases, how many were actually positive?”
Formula: Precision = TP / (TP + FP)
High precision = fewer false positives.
Recall	Measures the ability to find all positive instances. It answers: “Of all actual positive cases, how many were correctly identified?”
Formula: Recall = TP / (TP + FN)
High recall = fewer false negatives.

Key Difference:

Precision focuses on correctness of positive predictions.

Recall focuses on completeness of identifying positive cases.

Example (Spam Classifier):

High Precision: If the model marks an email as spam, it's almost certainly spam.

High Recall: The model catches almost every spam email, even if it misclassifies some real emails.

2. What is Cross-Validation, and why is it important in binary classification?
Cross-Validation is a technique to assess how well a model generalizes to unseen data by:

Splitting data into multiple parts (folds)

Training the model on some folds and testing it on the rest

Repeating this process multiple times

Averaging the results for a reliable performance estimate

Why it’s important:

✅ Reliable Performance Estimation: Avoids bias from a single train/test split

🔁 Reduces Overfitting: Helps detect models that memorize training data

⚙️ Hyperparameter Tuning: Enables more accurate tuning of model parameters

📊 Data Efficiency: Utilizes all available data, especially useful for small datasets

📉 Generalization Check: Ensures the model performs well on new, unseen data

Common Types of Cross-Validation:

k-Fold Cross-Validation: Standard method (e.g., 5 or 10 folds)

Stratified k-Fold: Ensures class distribution is maintained (especially for imbalanced datasets)

Leave-One-Out (LOOCV): Each instance is used once as a test set; ideal for very small datasets



## 📌 Conclusion
This project demonstrates the full cycle of building and deploying a binary classification model using logistic regression. It bridges machine learning with web-based interactivity using Streamlit, offering both technical insight and practical usability.

## 📬 Contact
For any suggestions or improvements, feel free to open an issue or submit a pull request.
