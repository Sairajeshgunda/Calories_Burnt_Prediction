
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#reading the dataset using pandas
calories_data=pd.read_csv("calories.csv")
calories_data.head()

exercise_data=pd.read_csv("exercise.csv")
exercise_data.head()


print("Missing values in calories_data['User_ID']:", calories_data['User_ID'].isnull().sum())
print("Missing values in exercise_data['User_ID']:", exercise_data['User_ID'].isnull().sum())

# If there are missing values, consider filling or dropping them
# calories_data['User_ID'].fillna(0, inplace=True)  # Example: Fill with 0
# exercise_data['User_ID'].fillna(0, inplace=True)  # Example: Fill with 0
calories_data['User_ID'].fillna(-1, inplace=True)
exercise_data['User_ID'].fillna(-1, inplace=True)

# Convert User_ID to integers to prevent potential issues during merge
calories_data['User_ID'] = calories_data['User_ID'].astype(int)
exercise_data['User_ID'] = exercise_data['User_ID'].astype(int)

# Check for duplicate User_IDs that might be causing issues
print("Duplicate User_IDs in calories_data:", calories_data['User_ID'].duplicated().sum())
print("Duplicate User_IDs in exercise_data:", exercise_data['User_ID'].duplicated().sum())

# If there are duplicates and you only want the first, drop then
# calories_data.drop_duplicates(subset='User_ID', keep='first', inplace=True)
# exercise_data.drop_duplicates(subset='User_ID', keep='first', inplace=True)

# Perform the merge
merged_data=pd.merge(calories_data, exercise_data, on='User_ID', how='outer', validate='many_to_many') #add validate for more info
merged_data.head()

#knowing the dimensions(rows, cols) of the dataset
calories_data.shape
exercise_data.shape

#knowing the datatypes involved int he dataset
calories_data.info()
exercise_data.info()

#making a statistical summarization of datasets
calories_data.describe()
exercise_data.describe()

#pivot the data for heatmap
plt.figure(figsize=(10,15))
heatmap_data=merged_data.pivot_table(index='Age', columns='Gender', values='Calories', aggfunc='mean')

#plotting the heatmap
sb.heatmap(heatmap_data, annot=True, cmap="seismic")
plt.show()

#Using a scatterplot for analysing data visually for Duration and Gender
sb.scatterplot(x='Duration', y='Gender', data=exercise_data)
plt.show()

#Using scatterplot for analysing data visually for weight and height
sb.scatterplot(x='Weight', y='Height', data=exercise_data)
plt.show()

#Making plots/graphs which represent the realtionship b/w 'features' and 'calories
features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = merged_data.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x, hue='Age', palette='Spectral')
plt.tight_layout()
plt.show()

#Making plots/graphs which represents the avg of height among boys and girls
#Also, the weight of the girls is lower than that of the boys.
#For the same average duration of workout calories burnt by men is higher than that of women
features = merged_data.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(merged_data[col])
plt.tight_layout()
plt.show()

#Making gender values; males=0, female=1 for better plotting
merged_data.replace({'male': 0, 'female': 1},
           inplace=True)
merged_data.head() #Printing Updated version
#Making a heatmap
plt.figure(figsize=(8, 8))
sb.heatmap(merged_data.corr() > 0.9,
           annot=True,
           cbar=False,
           cmap='coolwarm')
plt.show()

#Making a copy of merged_data named as merged_data_copy for deleting 'Weight', 'Height' from merged_data
merged_data_copy = merged_data.copy()  # Create a copy of merged_data
merged_data_copy.drop(columns=['Weight', 'Duration'], axis=1, inplace=True)  # Drop columns from the copy

#Importing sklearn for training the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Preapring data for ml model by splitting into training and validation sets
features = merged_data.drop(['User_ID', 'Calories'], axis=1)
target = merged_data['Calories'].values

X_train, X_val,\
    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.1,
                                      random_state=22)
X_train.shape, X_val.shape#((13500, 7), (1500, 7))

#Normalising the data for fast, stable and smooth training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
#To see the standarised values
print("X_train (standardized):\n", X_train)
print("\nX_val (standardized):\n", X_val)

#Importing the libs before training withb regression moodels
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

#Lets train the data using 5 different regression models and making predictions
from sklearn.metrics import mean_absolute_error as mae
models = [LinearRegression(), XGBRegressor(),
          Lasso(), RandomForestRegressor(), Ridge()]

for i in range(5):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')

    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    val_preds = models[i].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()