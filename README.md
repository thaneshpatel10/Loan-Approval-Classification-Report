
# Loan Default Prediction: A Comparative Model Analysis

This repository presents a comprehensive analysis of various machine learning models for predicting loan defaults.  We leveraged a dataset containing applicant demographics, financial history, and loan details to identify the most effective model for this critical classification task.

## Dataset

The dataset was taken from kaggle website. Key features include:

*   **Applicant Demographics:** `person_age`, `person_gender`, `person_education`, `person_home_ownership`
*   **Financial History:** `person_income`, `person_emp_exp`, `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`
*   **Loan Details:** `loan_amnt`, `loan_intent`, `loan_int_rate`, `loan_percent_income`
*   **Target Variable:** `loan_status` (1 for approved, 0 for rejected)

The dataset was partitioned into training (75%) and testing (25%) sets to ensure robust model evaluation.

## Methodology

Our analysis followed a structured approach:

1.  **Exploratory Data Analysis (EDA):**
2.  ** there were no missing values and dulicates in the dataset and also the data types were correct and correlation was found between the numerical features
3.  **Data Preprocessing:**
    *   **Multicollinearity Mitigation:**  Identified and addressed high correlation between `person_emp_exp` and `person_age` by removing `person_emp_exp` and 'person_age' *
    *   **Categorical Feature Encoding:** Applied one-hot encoding to convert categorical variables (`person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`) into a suitable numerical format.
    *   **Numerical Feature Scaling:** Standardized numerical features (`person_income`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`) using StandardScaler to ensure consistent feature scales.
4.  **Model Training and Evaluation:** Trained and evaluated the following classification models:
    *   K-Nearest Neighbors (KNeighborsClassifier)
    *   Support Vector Classifier (SVC)
    *   Decision Tree Classifier (DecisionTreeClassifier)
    *   Random Forest Classifier (RandomForestClassifier)
    *   Gradient Boosting Classifier (GradientBoostingClassifier)
    *   AdaBoost Classifier (AdaBoostClassifier)
    *   XGBoost Classifier (XGBClassifier)
   
5.  **Performance Metrics:**  Model performance was assessed using accuracy, precision, recall, F1-score, and confusion matrices.

## Results and Discussion

The following table summarizes the performance of the trained models on the test set:

| Model                     | Accuracy | Precision | Recall | F1-Score |
| ------------------------- | -------- | --------- | ------ | -------- |
| KNeighborsClassifier      | 0.893    | 0.79      | 0.72   | 0.75     |
| SVC                       | 0.915    | 0.85      | 0.76   | 0.80     |
| DecisionTreeClassifier    | 0.893    | 0.75      | 0.78   | 0.76     |
| RandomForestClassifier    | 0.927    | 0.88      | 0.78   | 0.83     |
| GradientBoostingClassifier | 0.923    | 0.87      | 0.77   | 0.82     |
| AdaBoostClassifier        | 0.903    | 0.78      | 0.79   | 0.79     |
| XGBClassifier             | 0.931    | 0.88      | 0.81   | 0.84     |

 "The XGBoost and Random Forest models demonstrated superior performance, achieving the highest accuracy and F1-scores.  The Decision Tree, while achieving perfect training accuracy, showed signs of overfitting, as evidenced by its lower test accuracy.  The lower recall scores across models suggest potential challenges in accurately identifying all default cases.  Further investigation into class imbalance and targeted model tuning may be beneficial.")*

## Conclusion

Based on our analysis, the **XGBoost Classifier** emerges as the top-performing model for loan default prediction, balancing accuracy, precision, and recall effectively.  The **Random Forest Classifier** also provides strong performance.  Future work may involve hyperparameter optimization, feature engineering, and exploring alternative models to further enhance predictive capabilities.




##The Dataset Was taken from kaggle Website
