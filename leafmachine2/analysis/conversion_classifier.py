import itertools
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot_decision_boundaries(X, y, model, feature_names, feature_indices, ax, resolution=1):
    # Setup marker generator and color map
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'lightgreen', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = np.percentile(X[:, 0], [1, 99])
    x2_min, x2_max = np.percentile(X[:, 1], [1, 99])
    
    # Increase resolution to reduce memory usage
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl if idx == 0 else "")

    ax.set_xlabel(feature_names[feature_indices[0]])
    ax.set_ylabel(feature_names[feature_indices[1]])
        
        

# Load the data from a CSV file
file_path = 'D:/D_Desktop/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily_PRED.csv' # Change this to your file path
data = pd.read_csv(file_path)


# Add the new column 'MP'
data['MP'] = (data['image_height'] * data['image_width']) / 1000000

# Select necessary columns
all_features = ['MP', 'conversion_mean', 'pooled_sd']
features = data[all_features]
labels = data['correct_conversion']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

X_train = X_train.fillna(0)
y_train = y_train.fillna(0)




do_create_DBV = True
if do_create_DBV:
    ### Decision Boundary Visualization
    # Create a grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Iterate over all unique pairs of features
    for ax, (i, j) in zip(axs.flatten(), itertools.combinations(range(len(all_features)), 2)):
        # Subsample your data for each pair of features
        # subsample_idx = np.random.choice(X_train.shape[0], size=100, replace=False)  # Adjust the size as needed
        # X_subsample = X_train.iloc[subsample_idx, [i, j]]  # Select only the two features
        # y_subsample = y_train.iloc[subsample_idx]

        # # Train a new model on these two features for visualization
        # rf_classifier.fit(X_train, y_train)

        # # Plotting decision boundaries for the current pair
        # plot_decision_boundaries(X_train.values, y_train, rf_classifier, all_features, [i, j], ax)


        # Extract the two features for the current pair
        X = X_train.iloc[:, [i, j]].values
        y = y_train.values

        # Train a new model on these two features for visualization
        rf_classifier.fit(X, y)

        # Plotting decision boundaries for the current pair
        plot_decision_boundaries(X, y, rf_classifier, all_features, [i, j], ax)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")



do_create_IP = True
if do_create_IP:
    ### Importances Plot
    # Get feature importances
    importances = rf_classifier.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), all_features)
    plt.xlabel('Relative Importance')
    plt.show()



do_create_PDP = True
if do_create_PDP:
    ### Partial Dependence Plots (PDPs)
    plot_partial_dependence(rf_classifier, X_train, all_features, grid_resolution=20)
    plt.show()



#############################
# Applying to new data
#############################
# Load new data
# new_data_file_path = 'D:/D_Desktop/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily.csv'  # Update this path
# new_data_file_path = 'D:/T_Downloads/LM2_Populus_subset_MEASUREMENTS.csv'
# new_data_file_path = 'D:/T_Downloads/LM2_Populus_subset_LANDMARKS_CFapplied.csv'
new_data_file_path = 'D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS.csv'
new_data = pd.read_csv(new_data_file_path)

new_data['MP'] = (new_data['image_height'] * new_data['image_width']) / 1000000

# Select the same features used for training
new_features = new_data[all_features]

# Replace NaN values with zeros
new_features_filled = new_features.fillna(0)

# Now use the filled data for prediction
new_predictions = rf_classifier.predict(new_features_filled)

# Append the predictions as a new column
new_data['correct_conversion'] = new_predictions

# Optionally, save the updated DataFrame to a new CSV file
# new_data.to_csv('D:/D_Desktop/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily_PRED.csv', index=False)  # Update the save path
new_data.to_csv('D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS_PRED.csv', index=False)






# # Filter the training set to include only rows where correct_conversion is 1
# train_indices = y_train[y_train == 1].index
# X_train_filtered = X_train.loc[train_indices, 'MP']  # Select only the 'MP' column
# y_train_filtered = data.loc[train_indices, 'conversion_mean']

# # Initialize the Random Forest Regressor
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the regressor on the filtered data (only 'MP' column)
# rf_regressor.fit(X_train_filtered.fillna(0).values.reshape(-1, 1), y_train_filtered.fillna(0))

# # Make predictions with the classifier
# new_predictions = rf_classifier.predict(new_features_filled)

# # Identify where the classifier predicts 0
mask = new_predictions == 0

# # Predict conversion_mean with the regressor for these instances
# # Only use 'MP' feature for prediction
# regression_predictions = rf_regressor.predict(new_features_filled.loc[mask, 'MP'].fillna(0).values.reshape(-1, 1))

# Filter the training set to include only rows where correct_conversion is 1
train_indices = y_train[y_train == 1].index
X_train_filtered = X_train.loc[train_indices, 'MP']  # Select only the 'MP' column
y_train_filtered = data.loc[train_indices, 'conversion_mean']

# Initialize Polynomial Features (choose the degree of the polynomial)
poly_degree = 2  # Example degree, you can change this
poly = PolynomialFeatures(degree=poly_degree)

# Transform 'MP' feature into polynomial features
X_train_poly = poly.fit_transform(X_train_filtered.fillna(0).values.reshape(-1, 1))

# Initialize the Linear Regression model
poly_regressor = LinearRegression()

# Train the regressor on the polynomial features
poly_regressor.fit(X_train_poly, y_train_filtered.fillna(0))

# ... [rest of the code remains unchanged until predicting with the regressor] ...

# Transform new_features for polynomial prediction
new_features_poly = poly.transform(new_features_filled.loc[mask, 'MP'].fillna(0).values.reshape(-1, 1))

# Predict conversion_mean with the polynomial regressor for these instances
regression_predictions = poly_regressor.predict(new_features_poly)

# Create a new column for predicted mean, initialize with NaN
new_data['pred_mean'] = np.nan

# Update pred_mean for instances where the classifier predicts 0
new_data.loc[mask, 'pred_mean'] = regression_predictions

# Optionally, save the updated DataFrame to a new CSV file
# new_data.to_csv('D:/D_Desktop/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily_PRED_MEAN_POLY.csv', index=False)  # Update the save path
new_data.to_csv('D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS_PRED_MEAN_POLY.csv', index=False)





# Save the model to a file
model_filename = 'rf_classifier.joblib'
dump(rf_classifier, model_filename)
# Load the model from the file
rf_classifier = load(model_filename)
# Use the loaded model to make predictions
loaded_predictions = rf_classifier.predict(X_test)

# Evaluate the loaded model
loaded_accuracy = accuracy_score(y_test, loaded_predictions)
print(f"Loaded Model Accuracy: {loaded_accuracy}")
