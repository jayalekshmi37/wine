
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Create a Dataframe and store it in a variable from the given dataset
df=pd.read_csv("https://raw.githubusercontent.com/jayalekshmi37/wine/refs/heads/main/WineQT.csv")
# Print the first 5 rows in the DataFrame
df.head()

#Values present in the quality attribute
set(df.quality)

#data types
df.dtypes

#number of rows and columns
print(df.shape)

#description of the dataset
df.describe()

# Create the plot
plt.figure(figsize=(15, 5))

# Plot each attribute with different colors and marker styles
plt.plot(df["fixed acidity"], "b-+", label="Fixed Acidity")
plt.plot(df["volatile acidity"], "g-+", label="Volatile Acidity")
plt.plot(df["citric acid"], "r-+", label="Citric Acid")
plt.plot(df["residual sugar"], "y-+", label="Residual Sugar")
plt.plot(df["chlorides"], "k-+", label="Chlorides")
plt.plot(df["free sulfur dioxide"], "m-+", label="Free Sulfur Dioxide")
plt.plot(df["total sulfur dioxide"], "c-+", label="Total Sulfur Dioxide")
plt.plot(df["density"], "b-o", label="Density")
plt.plot(df["pH"], "g-o", label="pH")
plt.plot(df["sulphates"], "r-o", label="Sulphates")
plt.plot(df["alcohol"], "y-o", label="Alcohol")
plt.plot(df["quality"], "k-o", label="Quality")

# Add a legend to the plot
plt.legend(loc="upper right")

plt.xlabel("Data points")
plt.ylabel("Attribute Value")

plt.show()

# Check for null values
print(df.isnull().sum())

# Create a box plot for each feature to identify outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=df.drop(columns=['Id']))  # Drop 'Id' if it is not relevant for outlier detection
plt.xticks(rotation=45)
plt.title('Box Plots of Wine Quality Dataset Features to Identify Outliers')
plt.show()

# Function to count the number of outliers in a column
def count_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
    return outliers.sum()

# Get all the numerical columns (excluding 'Id' if present)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
if 'Id' in numerical_columns:
    numerical_columns = numerical_columns.drop('Id')

# Count and display the number of outliers for each numerical column
for column in numerical_columns:
    num_outliers = count_outliers(column)
    print(f"Number of outliers in {column}: {num_outliers}")

# Function to replace outliers with the mean for a given column

def replace_outliers_with_mean(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                         df[column].mean(),
                         df[column])

def count_outliers_after_replacement(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Capping outliers to the lower and upper bounds
    df[column] = df[column].clip(lower_bound, upper_bound)

    # After capping, count outliers that are beyond the limits
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers.sum()

# Count and display the number of outliers after capping for each numerical column
for column in numerical_columns:
    num_outliers_after_replacement = count_outliers_after_replacement(df, column, multiplier=2.5)
    print(f"Number of outliers in {column} after replacement: {num_outliers_after_replacement}")

# Create a box plot for each feature after outlier removal
plt.figure(figsize=(15, 10))
for column in numerical_columns:
    count_outliers_after_replacement(df, column, multiplier=2.5)

sns.boxplot(data=df.drop(columns=['Id']))  # Drop 'Id' if it is not relevant for outlier detection
plt.xticks(rotation=45)
plt.title('Box Plots of Wine Quality Dataset Features After Outlier Treatment')
plt.show()

# Feature selection
X = df.drop('Id', axis=1)  # Selected all features except 'Id'
y=df['quality']
# Now X contains features for feature selection

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train-Test Split
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
predictions =pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Display the predictions
print(predictions.head())

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Get user input for wine features
fixed_acidity = float(input("Enter fixed acidity: "))
volatile_acidity = float(input("Enter volatile acidity: "))
citric_acid = float(input("Enter citric acid: "))
residual_sugar = float(input("Enter residual sugar: "))
chlorides = float(input("Enter chlorides: "))
free_sulfur_dioxide = float(input("Enter free sulfur dioxide: "))
total_sulfur_dioxide = float(input("Enter total sulfur dioxide: "))
density = float(input("Enter density: "))
pH = float(input("Enter pH: "))
sulphates = float(input("Enter sulphates: "))
alcohol = float(input("Enter alcohol: "))

# Create a new DataFrame with the user input, include a placeholder for 'quality'
new_wine = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol],
    'quality': [0]
})

# Predict the quality using the trained model
predicted_quality = rf_model.predict(new_wine)

# Print the predicted quality
print(f"Predicted Wine Quality: {predicted_quality[0]}")
