# Breast Cancer Prediction Project

## Overview

This project aims to develop a machine learning model for breast cancer prediction using a Random Forest classifier. The model is trained on a dataset containing various features extracted from breast cancer biopsies to classify tumors as benign or malignant.

## Dataset

Dataset -  [BreastCancer_prediction.csv](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) source Kaggle

The dataset used for this project contains the following features:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`

The target variable is `diagnosis`, which indicates whether a tumor is benign (B) or malignant (M).

## Installation

Clone GitRepository
```bash
git clone https://github.com/thillairam007/BreastCancerPrediction_CVIP-datascience
```

To run the Breast Cancer Prediction project, you need to install the required dependencies listed in the `requirements.txt` file. You can use `pip` to install them:

```bash
pip install -r requirements.txt
```
then, run the **breast_cancer_prediction.ipynb** File

# Random Forest Model Development

In this section, we'll outline the steps to develop a Random Forest model for breast cancer prediction using Python and scikit-learn.

## Steps

1. **Data Preprocessing:**

   - Load the breast cancer dataset.
   - Explore the dataset to understand its structure and features.
   - Check for missing data and handle it if necessary.
   - Encode categorical variables if applicable.
   - Split the dataset into training and testing sets.

2. **Feature Selection:**

   - Perform feature selection to choose the most relevant features for the prediction task. You can use techniques like feature importance from the Random Forest model.

3. **Model Initialization:**

   - Import the Random Forest classifier from scikit-learn.
   - Initialize the Random Forest classifier with hyperparameters like the number of estimators, maximum depth, and other relevant settings.

4. **Model Training:**

   - Fit the Random Forest classifier to the training data.
   - Train the model on the selected features.

5. **Model Evaluation:**

   - Evaluate the model's performance on the testing data.
   - Use metrics such as accuracy, precision, recall, F1-score

9. **Conclusion:**

   - Summarize the results and insights gained from the breast cancer prediction model.

## Example Code

Here's an example code snippet in Python to initialize and train a Random Forest classifier:

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)
```

## License
This project is licensed under the Apache 2.0  - see the [LICENSE](https://github.com/thillairam007/BreastCancerPrediction_CVIP-datascience/blob/main/LICENSE) file for details.


