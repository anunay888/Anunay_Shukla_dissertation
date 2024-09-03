#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score ,roc_curve, roc_auc_score , ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


# In[2]:


#Function which cleans the data
def preprocess_data(type1_path, type2_path):
    try:
        # Load the datasets
        type1_df=pd.read_csv(type1_path)
        type2_df=pd.read_csv(type2_path)
        
        # Strip any leading or trailing space whitespace and remove newline characters from column names
        type1_df.columns=type1_df.columns.str.strip().str.replace('\n', '')
        type2_df.columns=type2_df.columns.str.strip().str.replace('\n', '')
        # Print the cleaned columns name for verification
        print("Cleaned Type 1 columns:")
        print(repr(type1_df.columns.tolist()))
        
        print("\nCleaned Type 2 columns:")
        print(repr(type2_df.columns.tolist()))
        
        #Convert 'Registrations(numbers)' to numeric, coerce errors to NaN
        type1_df['Registrations (number)']=pd.to_numeric(type1_df['Registrations (number)'], errors='coerce')
        type2_df['Registrations (number)']=pd.to_numeric(type2_df['Registrations (number)'], errors='coerce')
        
        # Columns to normalise (i.e. convert from percentage to 0-1 scale)
        percentage_columns=[
            'Aged under 40(Percentage)','Aged 40 to 64(Percentage)' , 'Aged 65 to 79(Percentage)' , 
            'IMD most deprived(Percentage)' , 'IMD 2nd most deprived(Percentage)',
            'IMD least deprived(Percentage)' , 'White(Percentage)' , 'Asian or Asian British(Percentage)',
            'Black or Black British(Percentage)' , 'HBA1C_test(Percentage)' , 'Blood Pressure_test(Percentage)',
            'Cholestrol_test(Percentage)'
        ]
        
        #Normalise the percentage columns for both the datasets
        type1_df[percentage_columns]= type1_df[percentage_columns]/100
        type2_df[percentage_columns]= type2_df[percentage_columns]/100
        
        #Handle Missing data by filling with the mean, applied only to numeric values
        type1_df.fillna(type1_df.select_dtypes(include='number').mean(), inplace=True)
        type2_df.fillna(type2_df.select_dtypes(include='number').mean(), inplace=True)
        
        #Indicate succesful cleaning
        print("Cleaning completed succesfully!")
        
        return type1_df, type2_df
    except KeyError as e:
        print(f"KeyError: The column{e} was not found in DataFrame.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError:{e}.Please check the filepath.")
    except Exception as e:
        print(f"An Error occured during the basic analysis: {e}")
    return None,None
#This function does the exploratory data analysis
def Exploratory_Data_Analysis(df, dataset_name):
    try:
        print(f"EDA for{dataset_name} Dataset.")
        #Filter for positive values before applying logscale
        positive_df=df[df['Registrations (number)']>0]
        
        if not positive_df.empty:
            # Logarthmic Histogram of Registrations'
            plt.figure(figsize=(8,6))
            sns.histplot(positive_df['Registrations (number)'], kde=True, bins=20, color='blue', log_scale=(True, False))
            plt.title(f'Distribution of Registrations in {dataset_name} Dataset(logscale)')
            plt.xlabel('Number of Registrations( Log Scale)')
            plt.ylabel('Frequency')
            plt.show()
            
            # Boxplot of Registrations
            plt.figure(figsize=(8,6))
            sns.boxplot(x=df['Registrations (number)'],color='blue')
            plt.title(f'Boxplot of Registrations in{dataset_name} Dataset')
            plt.xlabel('Number of Registrations ')
            plt.show()
            
            # Correlation heatmap excluding ICB Code columns
            corr_columns= [col for col in df.columns if 'ICB Code' not in col]
            sns.heatmap(df[corr_columns].corr(), annot=True, fmt=".2f",cmap="coolwarm")
            plt.title(f"Correlation Heatmap for {dataset_name} Dataset(Excluding ICB Codes)")
            plt.show()
    except KeyError as e:
        print(f"KeyError: The column{e} was not found in the DataFrame.")
    except ValueError as e:
        print(f"ValueError: There was an issue with the data values:{e}")
    except Exception as e:
        print(f"An error occured during the basic analysis:{e}")
# providing the path of both the files
type1_path= "C:\\Users\\aloks\\OneDrive\\Documents\\type_1_dataset.csv"
type2_path= "C:\\Users\\aloks\\OneDrive\\Documents\\type 2_dataset.csv"
try:
    # Preprocess the data
    type1_preprocessed, type2_preprocessed= preprocess_data(type1_path, type2_path)
    #perform basic analysis
    if type1_preprocessed is not None:
        Exploratory_Data_Analysis(type1_preprocessed, "Type 1")
    if type2_preprocessed is not None:
        Exploratory_Data_Analysis(type2_preprocessed, "Type 2")
except Exception as e:
    print(f"An error occured in overall processing: {e}")
    
            
        
                      
        

        


# In[31]:


def save_preprocessed_datasets(type1_df, type2_df, type1_save_path, type2_save_path):
    try:
        #save the datasets 
        type1_df.to_csv(type1_save_path, index=False)
        type2_df.to_csv(type2_save_path, index=False)
        
        print(f"Type 1 dataset saved as '{type1_save_path}'")
        print(f"Type 2 dataset saved as '{type2_save_path}'")
    except Exception as e:{}
        print(f"An error occured while saving the datasets:{e}")
#provide the paths to save the datasets
type1_save_path='C://Users//aloks//OneDrive//Documents//preprocessed_type1_data.csv'
type2_save_path='C://Users//aloks//OneDrive//Documents//preprocessed_Type2.csv'
#call the function with the preproccesed datasets and save paths
save_preprocessed_datasets(type1_preprocessed, type2_preprocessed,type1_save_path,type2_save_path)


# In[4]:


def load_combined_dataset(file_path):
    try:
        #loading the file
        df=pd.read_csv(file_path)
        print(f"Dataset loaded from '{file_path}'")
        return df
    except FileNotFoundError:
        print(f"Error:The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error:The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: There was an error parsing the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occured while loading the dataset:{e}")
#checks for the class imbalance for the target feature
def analyze_class_imbalance(y, title):
    try:
        class_distribution= y.value_counts(normalize=True)*100
        print(f"Class distribution:\n{class_distribution}\n")
        
        #Plots the class distribution
        plt.figure(figsize=(8,6))
        class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f'Class Distribution {title}')
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.show()
    except TypeError as e:
              print(f"TypeError: {e}.Make sure the input 'y' is a Pandas Series.")
    except ValueError as e:
              print(f"ValueError: {e}. There might be an issue with the data types of the plotting.")
    except Exception as e:
              #catch all other exceptions
              print(f"An unexpected error occured during class imbalance analysis: {e}")
def remove_outliers(df):
    try:
        df_out=df.copy() #creating a copy of the dataset to avoid modifying the original one
        for column in df.columns:
            if df_out[column].dtype.kind in 'bifc': 
                    Q1=df_out[column].quantile(0.25)
                    Q3=df_out[column].quantile(0.75)
                    IQR= Q3-Q1
                    lower_bound= Q1-1.5*IQR
                    upper_bound=Q3+1.5*IQR
                    df_out=df_out[(df_out[column] >= lower_bound) & (df_out[column]<=upper_bound)]
        return df_out
    
    except KeyError as e:
        print(f"KeyError: {e}.One or more columns might be missing from the DataFrame")
    except TypeError as e:
        print(f"TypeError: {e}.Ensure all columns being processed are numeric.")
    except Exception as e:
        print(f"An unexpected error occured during outlier removal:{e}")
              
def select_high_correlation_features(df,target_column, threshold=0.3):
    try:
        #compute the correlation matrix
        corr_matrix=df.corr()
        
        #check if the target column is in the correlation matrix
        if target_column not in corr_matrix.columns:
              raise KeyError(f"Target column '{target_column}' not found in the DataFrame.")
              
        # select features with a correlation higher than the threshold
        high_corr_features= corr_matrix.index[corr_matrix[target_column].abs()>threshold].tolist()
        high_corr_features.remove(target_column) #removes the target column from the list
            
        print(f"Selected features with correlation > {threshold}: {high_corr_features}")
        return high_corr_features
    except KeyError as e:
        print(f"KeyError: {e}.One or more columns might be missing from the DataFrame")
    except ValueError as e:
        print(f"ValueError: There was an issue with the data values:{e}")
    except Exception as e:
        print(f"An unexpected error occured during outlier removal:{e}")

def machine_learning_pipeline(df, target_column= 'Registrations (number)', test_size=0.2, random_state=42, smote_flag=True, outlier_removal_flag=False, hyperparameter_optimization_flag=False,positive_correlation_flag=False,correlation_threshold=0.3):
    print("Starting the machine learning pipeline.......")
    #optionally remove the outliers
    if outlier_removal_flag:
        df = remove_outliers(df)
    
    #optionally select features based on correlation with the target variable
    if positive_correlation_flag:
        selected_features = select_high_correlation_features(df, target_column, correlation_threshold)
        X = df[selected_features]
    else:
        X = df.drop(columns= [target_column])
    
    y = (df[target_column] > df[target_column].mean()).astype(int)
    y_labels = y.replace({0: 'Low Risk', 1: 'High Risk'})
    
    #visualise class imbalance before SMOTE
    analyze_class_imbalance(y_labels, "Before SMOTE")
    
    # Data Splitting - Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size= test_size, random_state=random_state)
    
    #optionally apply SMOTE
    if smote_flag:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res_labels = smote.fit_resample(X_train, y_train)
        analyze_class_imbalance(pd.Series(y_train_res_labels), "After SMOTE")
    else:
        X_train_res, y_train_res_labels = X_train, y_train
        
    #convert the labels back to the numeric form for models like XGBoost
    y_train_res_numeric = y_train_res_labels.replace({'Low Risk': 0, 'High Risk': 1})
    y_test_numeric = y_test.replace({'Low Risk': 0, 'High Risk': 1})
    
    #Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Model selection - define the models
    models = {
        'Logistic Regression' : LogisticRegression(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}......")
        
        #optionally apply hyperparameter optimization
        if hyperparameter_optimization_flag:
            if model_name == 'Logistic Regression':
                param_distributions = {
                    'C': [0.1, 0.5, 1, 1.5, 2],
                    'max_iter' : [100, 200, 300]
                }
            elif model_name == 'Random Forest':
                param_distributions = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 3, 4],
                    'min_samples_leaf': [1, 2, 3],
                    'bootstrap': [True, False]
                    
                }
            elif model_name == 'XGBoost':
                param_distributions = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                
            randomized_search = RandomizedSearchCV(
                model,
                param_distributions=param_distributions,
                n_iter=10,
                cv=cv,
                verbose=1,
                random_state=random_state
            )
            randomized_search.fit(X_train_res, y_train_res_numeric if model_name == 'XGBoost' else y_train_res_labels)
            best_model = randomized_search.best_estimator_
            print(f"Best Parameters: {randomized_search.best_params_}")
        else:
            #cross-validation
            cross_val_scores = cross_val_score(model, X_train_res, y_train_res_numeric if model_name == 'XGBoost' else y_train_res_labels, cv=cv)
            print(f"cross-validation scores for {model_name}: {cross_val_scores}")
            best_model = model.fit(X_train_res, y_train_res_numeric if model_name == 'XGBoost' else y_train_res_labels)
        #Training and Testing
        y_train_pred = best_model.predict(X_train_res)
        y_test_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba")else None
        #convert predictions back to string labels for non-XGBoost models
        if model_name!= 'XGBoost':
            y_train_pred = pd.Series(y_train_pred).replace({0: 'Low Risk', 1: 'High Risk'})
            y_test_pred = pd.Series(y_test_pred).replace({0: 'Low Risk', 1: 'High Risk'})
        else:
            y_train_pred = pd.Series(y_train_pred)
            y_test_pred = pd.Series(y_test_pred)
        
        #ensuring both y_true and y_pred are of the same type
        if isinstance(y_train_pred.iloc[0], str):
            y_train_res_labels = y_train_res_labels.replace({0: 'Low Risk', 1: 'High Risk'})
            y_test = y_test.replace({0: 'Low Risk', 1:'High Risk'})
        else:
            y_train_res_labels = y_train_res_labels.replace({'Low Risk':0, 'High Risk':1})
            y_test = y_test.replace({'Low Risk':0, 'High Risk':1})
        
        #Evaluation
        train_accuracy = accuracy_score(y_train_res_labels, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test_numeric, y_proba) if y_proba is not None else None
        
        #Get the actual labels from y_test to avoid mismatch
        unique_labels= list(np.unique(y_test))
        
        #store the results
        results[model_name] = {
            'train_accuracy' : train_accuracy,
            'test_accuracy' : test_accuracy,
            'classification_report' : classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix' : confusion_matrix(y_test, y_test_pred, labels=unique_labels),
            'roc_auc' : roc_auc,
            'fpr' : None,
            'tpr' : None
        }
        
        if y_proba is not None:
            if model_name == 'XGBoost':
                fpr, tpr, _ = roc_curve(y_test_numeric, y_proba)
            else:
                fpr, tpr, _ = roc_curve(pd.get_dummies(y_test)['High Risk'], y_proba)
            results[model_name]['fpr'] =fpr
            results[model_name]['tpr'] =tpr
        
        # Output the results for each model.
        print(f"\nModel: {model_name}")
        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        print(f"ROC AUC Score: {roc_auc:.2f}" if roc_auc else "")
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
    # Automatically plot evaluation metrics
    plot_evaluation_metrics(results)
    
    return results

def plot_evaluation_metrics(results):
    metrics = ['train_accuracy', 'test_accuracy', 'precision' , 'recall', 'f1-score']
    model_names = list(results.keys())
    
    metric_values = {metric: [] for metric in metrics}
    
    for model_name in model_names:
        report = results[model_name]['classification_report']
        metric_values['train_accuracy'].append(results[model_name]['train_accuracy'])
        metric_values['test_accuracy'].append(results[model_name]['test_accuracy'])
        for metric in metrics[2:]:
            metric_values[metric].append(report['weighted avg'][metric])
    
    #plot bar charts for accuracy, precision, recall, and f1 score
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        ax.barh(model_names, metric_values[metric], color='skyblue')
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xlim(0, 1)
        
    plt.tight_layout()
    plt.show()
    
    #plot ROC AUC curves
    plt.figure(figsize=(10, 8))
    for model_name in model_names:
        if results[model_name]['roc_auc'] is not None:
            plt.plot(results[model_name]['fpr'], results[model_name]['tpr'], label=f"{model_name} (AUC = {results[model_name]['roc_auc']:.2f})")
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curves')
    plt.legend(loc='best')
    plt.show()
    
    #plot confusion matrices
    for model_name in model_names:
        confusion = results[model_name]['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Low Risk', 'High Risk'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()
       
def check_accuracy_difference(results, decimal_places=5):
    for model_name, metrics in results.items():

        train_accuracy = round(metrics['train_accuracy'], decimal_places)
        test_accuracy = round(metrics['test_accuracy'], decimal_places)
        accuracy_difference = train_accuracy - test_accuracy

        print(f"\nModel: {model_name}")
        print(f"Training Accuracy (rounded to {decimal_places} decimal places): {train_accuracy}")
        print(f"Testing Accuracy (rounded to {decimal_places} decimal places): {test_accuracy}")
        print(f"Accuracy Difference: {accuracy_difference:.{decimal_places}f}")
        if accuracy_difference == 0:
            print("No difference in accuracies upto the specified decimal places.")
        else:
            print(f"Difference detected: {accuracy_difference:.{decimal_places}f}")

def display_menu():
    print("Choose an option:")
    print("1. No Hyperparameter optimization (outliers not removed)")
    print("2. Hyperparameter optimization (outliers not removed)")
    print("3. No Hyperparameter optimization(outliers removed)")
    print("4. Hyperparameter optimization(outliers removed)")
    print("5. Positive Correlation features only(with or without other options)")
    choice = int(input("Enter your choice (1-5): "))
    return choice

def main_execution(df_combined):
    choice = display_menu()

    if choice == 1:
        results = machine_learning_pipeline(
        df=df_combined,
        target_column='Registrations (number)',
        smote_flag=True,
        outlier_removal_flag=False,
        hyperparameter_optimization_flag=False,
        positive_correlation_flag=False
        )
       
        check_accuracy_difference(results, decimal_places=5)
      
        
    elif choice == 2:
        results = machine_learning_pipeline(
        df=df_combined,
        target_column='Registrations (number)',
        smote_flag=True,
        outlier_removal_flag=False,
        hyperparameter_optimization_flag=True,
        positive_correlation_flag=False
        )
        
        check_accuracy_difference(results, decimal_places=5)
        
    elif choice == 3:
        results= machine_learning_pipeline(
        df=df_combined,
        target_column='Registrations (number)',
        smote_flag=True,
        outlier_removal_flag=True,
        hyperparameter_optimization_flag=False,
        positive_correlation_flag=False
        )
        
        check_accuracy_difference(results, decimal_places=5)
        
    elif choice == 4:
        df_combined_no_outliers = remove_outliers(df_combined)
        results = machine_learning_pipeline(
        df=df_combined_no_outliers,
        target_column='Registrations (number)',
        smote_flag=True,
        outlier_removal_flag=True,
        hyperparameter_optimization_flag=True,
        positive_correlation_flag=False
        )
        
        check_accuracy_difference(results, decimal_places=5)
    elif choice == 5:
        use_correlation_threshold = float(input("Enter the correlation threshold (e.g., 0.3): "))
        use_outliers = input("Do you want to remove outliers? (yes/no): ").lower() == "yes"
        use_optimization = input("Do you want to apply hyperparameter optimization? (yes/no): ").lower() == "yes"

        if use_outliers:
            df_combined_no_outliers = remove_outliers(df_combined)
            results = machine_learning_pipeline(
            df=df_combined_no_outliers,
            target_column="Registrations (number)",
            smote_flag=True,
            outlier_removal_flag=False,
            hyperparameter_optimization_flag=use_optimization,\
            positive_correlation_flag=True,
            correlation_threshold=use_correlation_threshold
                )
        else:
            print("Invalid choice! Please choose a valid option (1-5).")
            return

        

        # Check for accuracy difference
        check_accuracy_difference(results, decimal_places=5)

file_path='C://Users//aloks//OneDrive//Documents//type1_type2_preprocessed_combined.csv'
df_combined=load_combined_dataset(file_path)
main_execution(df_combined)
            


# In[ ]:




