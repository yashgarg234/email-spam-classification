# Email Spam Classification

## Project Domain / Category
**Data Science / Machine Learning**

## Introduction
Email is one of the most common forms of communication. With its growth, there's also an increase in unwanted emails, also known as **spam**. These spam messages often contain irrelevant, unsolicited content, and can pose a significant challenge in filtering them out effectively. 

In this project, we aim to build a machine learning model that classifies emails as either **spam** or **non-spam (ham)** using Python. We will apply several classification algorithms and compare their performance based on metrics such as **accuracy**, **time taken**, and **error rate**.

## Objective
The primary objective of this project is to develop a spam detection model using various machine learning techniques and determine which algorithm is best suited for email spam classification.

## Algorithms Used
- **Naive Bayes**
- **Naive Bayes Multinomial**
- **J48 (Decision Tree)**

## Features

### Data Collection
We will gather a dataset containing both spam and non-spam email messages.

### Data Pre-processing
Real-world data can be noisy and incomplete, so we'll clean and prepare the data by handling missing values and irrelevant features.

### Feature Selection
Feature selection helps in selecting the most relevant data for classification. We will use the **Best First Feature Selection** algorithm to identify the most significant features.

### Spam Filter Algorithms
We'll load the dataset, split it into training and test sets, and apply the spam filter algorithms to classify emails. The steps include:

1. **Data Handling**: Load the dataset and split it into training (70%) and testing (30%) sets.
2. **Summarization**: Summarize the properties in the training set to calculate probabilities for making predictions.
3. **Prediction**: Use the summaries to make predictions on the test set.
4. **Evaluation**: Evaluate the predictions by comparing them to the actual results and calculating the accuracy.

### Confusion Matrix
A **confusion matrix** will be used to analyze the performance of the classification model by providing detailed information on True Positives, False Positives, True Negatives, and False Negatives.

### Accuracy Calculation
We will compute the accuracy for all the algorithms and compare their performances to identify the best algorithm.

## Functional Requirements

The administrator will perform the following tasks:

1. **Collect Dataset**  
Gather email data containing spam and non-spam (ham) messages.

2. **Pre-processing**  
Clean and process the dataset by handling missing or noisy values.

3. **Feature Selection**  
Apply the **Best First Feature Selection** algorithm to select the most relevant features from the dataset.

4. **Apply Spam Filter Algorithms**  
Train the classification algorithms on the training data and test them on the test dataset.

5. **Train & Test Data**  
Split the dataset into training (70%) and testing (30%) sets for evaluating the performance of the models.

6. **Confusion Matrix**  
Create a confusion matrix to evaluate and visualize the performance of the classification model.

7. **Accuracy Calculation**  
Calculate the accuracy of each algorithm and compare the results.

## Project Setup

### Prerequisites
- **Python 3.x**
- Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tzmughal/spam-email-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd spam-email-classification
    ```


