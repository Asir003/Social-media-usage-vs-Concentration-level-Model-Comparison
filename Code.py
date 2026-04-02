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


        # Step 4:Handling duplicate values
        print("\n4. Detecting and removing duplicate entries...")
        duplicate_count = self.df.duplicated().sum()

        if duplicate_count > 0:
            print(f"⚠️ Found {duplicate_count} duplicate rows. Removing them...")
            self.df = self.df.drop_duplicates()
            print("✅ Duplicates removed successfully.")
        else:
            print("✅ No duplicate rows found.")

        # Step 5:Creating focus time categories for classification
        print("\n5.Creating focus time categories for classification...")

        # Map the original text to simpler labels
        focus_mapping = {
            "Less than 15 minutes": "Short",
            "15–30 minutes": "Medium",
            "More than 30 minutes": "Long"
        }

        self.df['focus_category'] = self.df['focus_time'].map(focus_mapping)

        print("  focus_time categories created successfully!")
        print(" - Categories:", self.df['focus_category'].unique())

        # Step 6:Encode all features
        print("\n6. Encoding categorical variables...")
        X = self.df[['daily_hours', 'check_while_study', 'notification_distraction', 'use_in_class']]
        y = self.df['focus_category']

        X_encoded = X.copy()
        for col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
            print(f"✅ Encoded '{col}': {len(le.classes_)} unique values")

        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"✅ Encoded target 'focus_category': {len(self.target_encoder.classes_)} classes")

        """
        print(self.label_encoders['daily_hours'].classes_)
        print(X_encoded.head())
        print(y_encoded[:10])
        """
        self.X_processed = X_encoded
        self.y_processed =  y_encoded
        self.target_classes = list(self.target_encoder.classes_)

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"  - Final features: {list(X_encoded.columns)}")
        print(f"  - Feature count: {X_encoded.shape[1]}")
        print(f"  - Sample count: {X_encoded.shape[0]}")

    def perform_eda(self):
        print("\n" + "=" * 60)
        print("Performing Exploratory Data Analysis (EDA)...")
        print("=" * 60)
        
        # Summary Statistics
        print("\n1. Summary Statistics:")
        print(self.df.describe(include='all'))
        
        # Configure layout for 2 plots
        fig = plt.figure(figsize=(12, 6))
        
        # 1. Bar Chart: Count of Focus Categories
        ax1 = plt.subplot(1, 2, 1)
        sns.countplot(data=self.df, x='focus_category', order=['Short', 'Medium', 'Long'], palette='Set2', ax=ax1)
        ax1.set_title('Bar Chart: Focus Categories Distribution', fontweight='bold')
        
        # 2. Pie Chart: Use in Class
        ax2 = plt.subplot(1, 2, 2)
        class_counts = self.df['use_in_class'].value_counts()
        ax2.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        ax2.set_title('Pie Chart: Use Social Media in Class', fontweight='bold')
        
        plt.tight_layout()
        print("✓ Displaying EDA Visualizations (Bar Chart and Pie Chart)...")
        print("  → Close the plot window to continue with the ML training")
        self._show_and_close(fig)

    def split_data(self, test_size=0.2, random_state=42):
        print("\n" + "=" * 60)
        print("Splitting Data...")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_processed,
            self.y_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_processed
        )
        
        print(f"  Data split successfully!")
        print(f"  - Training set: {self.X_train.shape[0]} samples ({1-test_size:.0%})")
        print(f"  - Testing set: {self.X_test.shape[0]} samples ({test_size:.0%})")
        print(f"  - Features: {self.X_train.shape[1]}")

    def train_logistic_regression(self):
      
        print("\n" + "=" * 60)
        print("Training Logistic Regression Classifier...")
        print("=" * 60)
        
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'saga']
        }
        
        base_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
        )
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    def train_random_forest(self):
        
        print("\n" + "=" * 60)
        print("Training Random Forest Classifier...")
        print("=" * 60)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['Random Forest Classifier'] = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")


    def train_all_models(self):
        
        self.train_logistic_regression()
        self.train_random_forest() 

    def evaluate_models(self):
        
        print("\n" + "=" * 60)
        print("Evaluating Models...")
        print("=" * 60)
        
        metrics = {}
        label_names = self.target_classes if self.target_classes else list(self.target_encoder.classes_)
        num_labels = len(label_names)
        if num_labels == 0:
            raise ValueError("Target classes are not defined. Ensure preprocessing has been completed before evaluation.")
        
        for name, model in self.models.items():
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
            conf_mat = confusion_matrix(self.y_test, y_test_pred, labels=np.arange(num_labels))
            
            metrics[name] = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Confusion Matrix': conf_mat,
                'y_test_pred': y_test_pred
            }
            
            print(f"\n{name}:")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Precision (weighted): {precision:.4f}")
            print(f"  Recall (weighted): {recall:.4f}")
            print(f"  F1-Score (weighted): {f1:.4f}")
        
        self.model_scores = metrics
        
        # Select best model based on Test Accuracy
        best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['Test Accuracy'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print("\n" + "=" * 60)
        print(f" Best Model: {self.best_model_name}")
        print("=" * 60)
        print(f"  Test Accuracy: {metrics[best_model_name]['Test Accuracy']:.4f}")
        print(f"  Precision (weighted): {metrics[best_model_name]['Precision']:.4f}")
        print(f"  Recall (weighted): {metrics[best_model_name]['Recall']:.4f}")
        print(f"  F1-Score (weighted): {metrics[best_model_name]['F1 Score']:.4f}") 


    @staticmethod
    def _show_and_close(fig):
       
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        plt.close(fig)

    def plot_confusion_matrix(self, model_name):
        
        if model_name not in self.model_scores:
            print(f"  ⚠ Warning: Model '{model_name}' not found in model_scores")
            return

        conf_mat = self.model_scores[model_name]['Confusion Matrix']
        if conf_mat is None:
            print(f"  ⚠ Confusion matrix not available for {model_name}")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=self.target_classes,
            yticklabels=self.target_classes,
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')

        plt.tight_layout()
        print(f"✓ Displaying confusion matrix for {model_name}...")
        print(f"  → Close the plot window to view the next visualization")
        self._show_and_close(fig)

    def plot_roc_curve(self, model_name):
        
        if model_name not in self.models:
            print(f"   Warning: Model '{model_name}' not found in models")
            return

        model = self.models[model_name]
        if not hasattr(model, "predict_proba"):
            print(f"   Model '{model_name}' does not support probability estimates required for ROC curve.")
            return

        y_score = model.predict_proba(self.X_test)
        classes = self.target_classes if self.target_classes else self.target_encoder.classes_
        y_test_binarized = label_binarize(self.y_test, classes=np.arange(len(classes)))

        fig, ax = plt.subplots(figsize=(10, 6))
        color_palette = sns.color_palette("Set2", len(classes))

        for idx, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, idx], y_score[:, idx])
            auc_score = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=color_palette[idx],
                linewidth=2,
                label=f"{class_name} (AUC = {auc_score:.3f})"
            )

        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Chance')
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        print(f"✓ Displaying ROC curve for {model_name}...")
        print(f"  → Close the plot window to view the next visualization")
        self._show_and_close(fig)

    def visualize_results(self):

        print("\n" + "=" * 60)
        print("Creating Visualizations...")
        print("=" * 60)
        
        # Step 1: Show individual plots for each model sequentially
        print("\n1. Displaying Logistic Regression classification visualization...")
        self.plot_confusion_matrix('Logistic Regression')
        self.plot_roc_curve('Logistic Regression')
        
        print("\n2. Displaying Random Forest classification visualization...")
        self.plot_confusion_matrix('Random Forest Classifier')
        self.plot_roc_curve('Random Forest Classifier')


         # Step 2: Create final comparison figure with classification metrics
        print("\n4. Creating classification metrics comparison visualization...")
        models = list(self.model_scores.keys())
        accuracies = [self.model_scores[m]['Test Accuracy'] for m in models]
        precisions = [self.model_scores[m]['Precision'] for m in models]
        recalls = [self.model_scores[m]['Recall'] for m in models]
        f1_scores = [self.model_scores[m]['F1 Score'] for m in models]

        x = np.arange(len(models))
        width = 0.2
        
        accuracy_color  = "#6A5ACD"  
        precision_color = "#FFB347"  
        recall_color    = "#77DD77"  
        f1_color        = "#779ECB"  

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy',color=accuracy_color)
        ax.bar(x - 0.5*width, precisions, width, label='Precision',color=precision_color)
        ax.bar(x + 0.5*width, recalls, width, label='Recall',color=recall_color)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score',color=f1_color)

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        print(" Visualization saved as 'model_performance_comparison.png'")
        
        # Create a metrics comparison table
        print("\n" + "=" * 60)
        print("Model Performance Summary")
        print("=" * 60)
        summary_df = pd.DataFrame({
            'Model': models,
            'Test Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores
        })
        summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
        print(summary_df.to_string(index=False))
        
        self._show_and_close(fig)

    def save_best_model(self, filename='best_car_price_classifier.joblib'):
       
        print("\n" + "=" * 60)
        print("Saving Best Model...")
        print("=" * 60)
        
        # Create a dictionary with model and preprocessors
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': list(self.X_train.columns),
            'model_name': self.best_model_name,
            'target_encoder': self.target_encoder,
            'target_classes': self.target_classes
        }
        
        joblib.dump(model_package, filename)
        print(f"  Best model saved as '{filename}'")
        print(f"  - Model: {self.best_model_name}")
        print(f"  - Includes: model, scaler, label encoders, target encoder, and feature columns")

    def run_complete_pipeline(self):
        
        print("\n" + "=" * 60)
        print("Social-media-usage-vs-Concentration-level-Model-Comparison - MACHINE LEARNING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess data
        self.preprocess_data()

        # Step 3: Exploratory Data Analysis
        self.perform_eda()

        # Step 4: Split data
        self.split_data()

        # Step 5: Train models
        self.train_all_models()

        # Step 6: Evaluate models
        self.evaluate_models()

        # Step 7: Visualize results
        self.visualize_results()

        # Step 8: Save best model
        self.save_best_model()


def main():
    
    # Initialize predictor
    predictor = Concentration(csv_file='Social.csv')
    
    # Run complete pipeline
    predictor.run_complete_pipeline()


if __name__ == "__main__":
    main()