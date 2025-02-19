# Data-Science-Codealpha-Internship-Task
Data Science Codealpha Internship Task
# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Step 2: Load the Iris dataset
# If you have a CSV file, change the path accordingly
iris = pd.read_csv("IRIS.csv")

# Step 3: Explore the dataset
print("First 5 rows of the dataset:")
print(iris.head())

# Descriptive statistics
print("\nDescriptive statistics:")
print(iris.describe())

# Unique target labels (species)
print("\nTarget Labels:", iris["species"].unique())

# Step 4: Visualize the data
# Scatter plot to visualize the relationship between sepal width and sepal length
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# Step 5: Prepare data for model training
# Separate features (X) and target labels (y)
X = iris.drop("species", axis=1)  # Features
y = iris["species"]  # Target labels

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Train the KNN classifier
# Initialize KNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 8: Evaluate the model
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Hyperparameter Tuning (Optional)
# If you want to experiment with different values of k, use GridSearchCV
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Display the best value of k (number of neighbors)
print("\nBest number of neighbors (k) from GridSearchCV:", grid_search.best_params_["n_neighbors"])

# Step 10: Make a new prediction with the best model (optional)
# Example new measurement: [sepal_length, sepal_width, petal_length, petal_width]
x_new = np.array([[5.1, 3.5, 1.4, 0.2]])

# Use the best model to predict the class
best_knn = grid_search.best_estimator_
prediction = best_knn.predict(x_new)

print("\nPrediction for new sample (5.1, 3.5, 1.4, 0.2):", prediction)
