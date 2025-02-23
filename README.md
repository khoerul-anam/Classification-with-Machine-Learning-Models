# 🎗️ **Breast Cancer Classification with Machine Learning Models** 🤖

## 📚 **Project Description**
This project aims to classify breast cancer tumors as **malignant** (cancerous) or **benign** (non-cancerous) using a variety of **machine learning models**. The dataset used for this project is the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) from [Scikit-learn](https://scikit-learn.org/stable/index.html), which contains 30 features describing various properties of cell nuclei present in breast cancer biopsies. The goal is to explore and compare the performance of different classification algorithms, including:

- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/svm.html)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

## 🎯 **Goals**
The main goals of this project are:
1. **Build and train classification models** using popular machine learning algorithms to predict breast cancer malignancy.
2. **Evaluate the performance** of each model using accuracy as the primary metric.
3. **Compare the strengths and weaknesses** of each algorithm and choose the best-performing model for this classification task.
4. **Visualize the model performance** through bar charts to make the comparison clearer and easier to understand.
5. **Explore feature importance** with Random Forest and discuss its implications in real-world cancer diagnosis.

## ⚙️ **Algorithms Used**
Here are the models applied to solve this classification task:

1. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) : An ensemble learning method that uses multiple decision trees to improve accuracy and reduce overfitting.
2. [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) : A simple, non-parametric algorithm that classifies a data point based on the majority label of its neighbors.
3. [Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/svm.html) : A powerful classifier that seeks to find the optimal hyperplane separating different classes.
4. [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) : A probabilistic model based on Bayes' Theorem, assuming that features are independent given the class label.

## 🧠 **Insight**
From the results of the project, some key insights include:
- All four models ([Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [SVM](https://scikit-learn.org/stable/modules/svm.html), [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), and [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) performed well in terms of classification accuracy.
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) showed the highest accuracy at **0.97**.
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [SVM](https://scikit-learn.org/stable/modules/svm.html), [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), and [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) all achieved an accuracy of **0.96**.
- The results suggest that although the models performed similarly, **Naive Bayes** slightly outperformed the others, likely due to its ability to handle probabilistic relationships efficiently in the dataset.

## 🛠️ **Dependencies**
To run this project locally, you need to install the following Python dependencies:

- 📚 [Scikit-learn](https://scikit-learn.org/stable/index.html): A machine learning library providing tools for data mining and data analysis.
- 📊 [matplotlib](https://matplotlib.org/): A plotting library for creating static, animated, and interactive visualizations in Python.
- 🔢 [numpy](https://numpy.org/): A package for scientific computing with Python, providing support for arrays and matrices.
- 📋 [pandas](https://pandas.pydata.org/): A data manipulation and analysis library that provides data structures like DataFrames.

