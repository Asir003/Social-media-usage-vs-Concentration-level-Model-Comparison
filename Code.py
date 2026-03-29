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

        self.valid_options_dict = {
            'daily_hours': [
                "Less than 1 hour", "1–2 hours", "2–4 hours", "4–6 hours", "More than 6 hours"
            ],
            'check_while_study': [
                "Never", "Rarely", "Sometimes", "Often", "Almost always"
            ],
            'notification_distraction': [
                "Not at all", "A little", "Sometimes", "Very much"
            ],
            'use_in_class': [
                "Never", "Sometimes", "Often", "Always"
            ],
            'focus_time': [
                "Less than 15 minutes", "15–30 minutes", "More than 30 minutes"
            ]
        }

        self.required_columns = [
            'daily_hours', 'check_while_study', 'notification_distraction', 'use_in_class', 'focus_time'
        ]
        
        self.categorical_columns = ['daily_hours', 'check_while_study', 'notification_distraction', 'use_in_class', 'focus_time']


    def load_data(self):
      
        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"  Dataset loaded successfully!")
            
            # Rename Columns
            self.df.rename(columns={
            '1.How many hours per day do you usually use social media during a regular academic day (when classes are running)?': 'daily_hours',
            '2. How often do you check social media while studying?': 'check_while_study',
            '3. Do social media notifications distract you while studying?': 'notification_distraction',
            '4. Do you use social media during class?': 'use_in_class',
            '5. How long can you study continuously without distraction?': 'focus_time'
            }, inplace=True)
            print(f"  Rename Columns successfully!")
            print(f"  - Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"  - Columns: {list(self.df.columns)}")
        except FileNotFoundError:
            print(f"✗ Error: File '{self.csv_file}' not found!")
            raise
        except Exception as e:
            print(f"✗ Error loading file: {str(e)}")
            raise
    
    def preprocess_data(self):
       
        print("\n" + "=" * 60)
        print("Preprocessing Data...")
        print("=" * 60)
        
        # Step 1: Keep only required columns
        print("\n1. Selecting required columns...")
        missing_columns = [col for col in self.required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"    Warning: Missing columns: {missing_columns}")
        
        # Drop columns not in required list
        columns_to_drop = [col for col in self.df.columns if col not in self.required_columns]
        if columns_to_drop:
            # Also drop the first unnamed index column if it exists
            if self.df.columns[0] == 'Unnamed: 0' or self.df.columns[0] == '':
                columns_to_drop.append(self.df.columns[0])
            self.df = self.df.drop(columns=columns_to_drop, errors='ignore')
            print(f"    Dropped {len(columns_to_drop)} unnecessary columns: {columns_to_drop}")
        
         # Ensure all required columns exist
        available_columns = [col for col in self.required_columns if col in self.df.columns]
        self.df = self.df[available_columns]
        print(f"    Using {len(available_columns)} columns for modeling")

         # Step 2: Handle missing values
        print("\n2. Handling missing values...")
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print("  Missing values per column:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    - {col}: {count} ({count/len(self.df)*100:.2f}%)")

            # For categorical columns: use mode
            for col in self.categorical_columns:
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            print("    Missing values filled")
        else:
            print("    No missing values found")
 
         # Step 3:Check is there any others values and remove the leading and trailing spaces
        print("\n3. Checking required columns for unexpected values and cleaning spaces...")
        for col, valid_options in self.valid_options_dict.items():
        # Strip spaces
            self.df[col] = self.df[col].astype(str).str.strip()

            # Find unexpected values
            mask_invalid = ~self.df[col].isin(valid_options)

            if mask_invalid.any():
                # Get the mode (most frequent value) of the valid values in the column
                mode_value = self.df.loc[~mask_invalid, col].mode()[0]  # most common valid value
                # Replace unexpected values with mode
                self.df.loc[mask_invalid, col] = mode_value
                print(f"⚠️ Column '{col}' had {mask_invalid.sum()} unexpected values -> replaced with mode '{mode_value}'")
            else:
                print(f" Column '{col}' is clean. No unexpected values found.")
        
        
        print("✅ All required columns cleaned and validated with mode.")
        
    


    def run_complete_pipeline(self):
        
        print("\n" + "=" * 60)
        print("Social-media-usage-vs-Concentration-level-Model-Comparison - MACHINE LEARNING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess data
        self.preprocess_data()


def main():
    
    # Initialize predictor
    predictor = Concentration(csv_file='Social.csv')
    
    # Run complete pipeline
    predictor.run_complete_pipeline()


if __name__ == "__main__":
    main()