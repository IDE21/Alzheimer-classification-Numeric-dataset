# Alzheimer-classification-Numeric-dataset

## **Overview**

In this project, we aim to develop a robust machine learning model to predict Alzheimer’s disease status based on clinical data. The dataset includes multiple clinical features like cognitive scores, brain volume measures, age, and education, among others. The main goal is to classify individuals into distinct groups: *Non-Demented*, *Very Mild Demented*, and *Mild Demented*, based on their features.

We leverage a variety of powerful machine learning models, including **Random Forest**, **XGBoost**, **LightGBM**, and **CatBoost**, to build and evaluate models. Furthermore, we combine these models into an **ensemble method** to boost performance. 

We also integrate advanced interpretability tools like **SHAP** and **LIME** to better understand how the model makes decisions and ensure the predictions are explainable, which is crucial for applications in healthcare.

## **Dataset**

The dataset used in this project contains clinical data on Alzheimer's disease and consists of 10 columns:
- **Age**: The age of the participant.
- **EDUC**: The education level (in years).
- **SES**: Socio-economic status.
- **MMSE**: Mini-Mental State Examination score, a cognitive screening test.
- **CDR**: Clinical Dementia Rating scale, which quantifies dementia severity.
- **eTIV**: Estimated Total Intracranial Volume.
- **nWBV**: Normalized Whole Brain Volume.
- **ASF**: Atlas Scaling Factor.
- **Group**: The target variable, representing the disease status (Non-Demented, Very Mild Demented, Mild Demented).

## **Key Steps in the Project**

### 1. **Data Preprocessing**
   - **Handling Missing Values**: Missing values are detected and filled using either the mean (for numerical columns) or mode (for categorical columns).
   - **Encoding Categorical Variables**: Categorical variables are transformed into numerical form using **LabelEncoder**.
   - **Feature Scaling**: The numerical features are standardized using **StandardScaler**, which helps the machine learning models perform better by ensuring all features are on the same scale.

### 2. **Modeling**
   We used the following machine learning models to train and evaluate the dataset:
   
   - **Random Forest Classifier**: A powerful ensemble method that builds many decision trees and outputs the mode of their predictions.
   - **XGBoost Classifier**: An optimized gradient boosting algorithm that often performs better on tabular data.
   - **LightGBM Classifier**: A gradient boosting framework designed for efficiency and scalability.
   - **CatBoost Classifier**: Another gradient boosting method that performs well with categorical features.
   - **Voting Classifier (Ensemble Learning)**: A combination of multiple models to improve performance by voting on the best prediction.

### 3. **Model Evaluation**
   - **Classification Metrics**: Accuracy, Precision, Recall, F1-score, and ROC-AUC scores are used to evaluate model performance.
   - **Confusion Matrix**: A confusion matrix is plotted to visually assess the classification results.

### 4. **Model Interpretability**
   - **SHAP (SHapley Additive exPlanations)**: Used to explain the global and local predictions made by the machine learning models. This helps us understand which features are most important in predicting Alzheimer’s status.
   - **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local explanations for individual predictions, showing how each feature contributes to the predicted outcome.

### 5. **Visualization**
   - **ROC Curve**: A Receiver Operating Characteristic (ROC) curve is plotted for each class to evaluate the classifier's performance.
   - **Feature Importance**: Visualizes the most important features in the model using SHAP values.
   - **SHAP Summary Plot**: This plot helps in understanding the overall model behavior by summarizing the impact of all features on predictions.

### 6. **Model Saving**
   - **Model Persistence**: The best-performing models are saved using **joblib** for later use, allowing you to load them for predictions in the future without retraining.

---

## **Features and Results**

### **Multiple Models**
   - The project compares **Random Forest**, **XGBoost**, **LightGBM**, and **CatBoost**, leveraging their respective strengths in handling structured data and achieving high accuracy.

### **Ensemble Learning**
   - By combining multiple models using a **VotingClassifier**, the project achieves better accuracy and robustness compared to using a single model.

### **Model Interpretability**
   - **SHAP** values explain the global model decision-making process, while **LIME** provides insights into individual predictions, making the model transparent and trustworthy.
   - These explainability tools are particularly important in healthcare applications, where understanding model predictions can directly influence clinical decision-making.

### **Evaluation Metrics**
   - In addition to accuracy, other important metrics such as **precision**, **recall**, **F1-score**, and **AUC** provide a comprehensive view of model performance.
   - The **ROC curve** is also plotted to assess how well the model distinguishes between the classes.

---

## **Installation**

To run the project, install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes libraries such as:
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **scikit-learn**: For building machine learning models and evaluating them.
- **xgboost**, **lightgbm**, **catboost**: For boosting algorithms.
- **shap**: For model interpretability.
- **lime**: For local explainability.
- **matplotlib**, **seaborn**: For visualizations.

---

## **Usage**

Once the dependencies are installed, you can run the code in sequence:
1. **Data Preprocessing**: Load and preprocess the dataset.
2. **Model Training**: Train various classifiers such as Random Forest, XGBoost, LightGBM, and CatBoost.
3. **Evaluation**: Assess the model performance using metrics like accuracy, ROC-AUC, confusion matrix, and F1 score.
4. **Visualization**: Generate ROC curves, SHAP plots, and feature importance visualizations.
5. **Model Saving**: Save the trained models for future predictions.

```python
# Example: Running the trained models
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## **Future Work**

- **Hyperparameter Tuning**: Fine-tuning hyperparameters to improve model performance further.
- **Deep Learning Models**: Incorporating deep learning models like **neural networks** to compare results with traditional machine learning models.
- **Model Deployment**: Deploying the model into a web application for real-time Alzheimer’s disease prediction.
- **Multimodal Data**: Incorporating additional data sources, such as imaging or genetic information, to enhance predictions.

---

## **Conclusion**

This project demonstrates the power of machine learning models in predicting Alzheimer’s disease based on clinical data. The use of ensemble learning techniques and model interpretability tools like **SHAP** and **LIME** ensures that the models are not only accurate but also explainable, making them suitable for real-world applications in healthcare.
