# ML Lab - MCA521-4

Machine Learning lab work for MCA521-4, IV MCA (2026).

**Praneeth M — 2547142**

`main` holds every lab. Each lab also has its own branch (`lab1`, `lab2`, ... `lab10`) containing
only that lab, as required by the submission guidelines.

## Labs

| Lab | Topic | Dataset | Folder |
|---|---|---|---|
| 1 | Data preprocessing and visualisation | India city AQI + crop production | `lab_1_2/` |
| 2 | Data exploration and inferences | India city AQI + crop production | `lab_1_2/` |
| 3 | Simple linear regression with Scikit-learn, manual OLS and parameter saving | Class student survey | `lab3/` |
| 4 | KNN classification and regression vs classification evaluation metrics | Breast Cancer Wisconsin (Diagnostic) | `lab4/` |
| 5 | Linear regression through gradient descent | UCI Student Performance | `lab5/` |
| 6 | Logistic Regression vs KNN classifiers | Breast Cancer Wisconsin (Diagnostic) | `lab6/` |
| 7 | Decision Tree classifier | Iris | `lab7/` |
| 8 | Naive Bayes classifier | Play Tennis | `lab8/` |
| 9 | Support Vector Machine and PCA (with LDA extra credit) | Breast Cancer Wisconsin + Wine | `lab9/` |
| 10 | Learning XOR with an MLP in Keras, PyTorch and TensorFlow | XOR truth table | `lab10/` |

`distance_matrix/` holds the in-class distance matrix activity.

## Contents of each folder

The question document, the solution notebook (`.ipynb`) with all outputs, an HTML export, a PDF
export, and the dataset used.

## Running the notebooks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
jupyter lab
```

Lab 10 additionally needs `tensorflow` and `torch`.
