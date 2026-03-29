import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


import joblib
import matplotlib.pyplot as plt
import seaborn as sns


try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")

class Concentration:

    def __init__(self, csv_file='Social.csv'):
       
        self.csv_file = csv_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.target_classes = None
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None


def main():
    
    # Initialize predictor
    predictor = Concentration(csv_file='Social.csv')
    
    # Run complete pipeline
    predictor.run_complete_pipeline()


if __name__ == "__main__":
    main()