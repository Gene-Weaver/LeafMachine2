import itertools, os, inspect, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

from joblib import dump, load
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
# from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
sys.path.append(parentdir)
from leafmachine2.machine.general_utils import validate_dir

class RandomForestModel:
    def __init__(self, data_path, features):
        self.data_path = data_path
        self.features = features
        self.data = pd.read_csv(data_path)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.prepare_data()

    def prepare_data(self):
        self.data['MP'] = (self.data['image_height'] * self.data['image_width']) / 1000000

        # Selecting the specified features and the target
        self.X = self.data[self.features]
        self.y = self.data['correct_conversion']

        # Splitting the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42)

        # Fill missing values with zero (consider appropriate imputation methods)
        self.X_train.fillna(0, inplace=True)
        self.y_train.fillna(0, inplace=True)
        self.X_test.fillna(0, inplace=True)

    def plot_decision_boundaries(self, feature_indices, ax, resolution=100, poly_model=None):
        # Extract the features based on indices
        feature_names = [self.features[i] for i in feature_indices]
        X = self.X_train[feature_names].values
        y = self.y_train.values

        markers = ('o', 'x', 's', '^', 'v')
        colors = ('red', 'lightgreen', 'blue', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        
        # Create a grid for plotting
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, resolution),
                               np.linspace(x2_min, x2_max, resolution))
        Z = self.model.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)

        ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=0.8, c=colors[idx],
                       marker=markers[idx], label=f"Class {cl}")

        # Plot polynomial line if appropriate
        if poly_model and 'MP' in feature_names:
            self.plot_poly_line(ax, poly_model, version='initial', color='r--')
            self.plot_poly_line(ax, poly_model, version='final', color='k--')

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

    def plot_poly_line(self, ax, poly_model, version, color):
        if version == 'initial':
            # Example for plotting a polynomial regression line on the decision plot
            x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 400)
            x_poly = PolynomialFeatures(degree=poly_model.poly_degree).fit_transform(x_range.reshape(-1, 1))
            y_poly = poly_model.initial_poly_model.predict(x_poly)
            ax.plot(x_range, y_poly, color, label="Poly Fit Initial")
            ax.legend()
        elif version == 'final':
            # Example for plotting a polynomial regression line on the decision plot
            x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 400)
            x_poly = PolynomialFeatures(degree=poly_model.poly_degree).fit_transform(x_range.reshape(-1, 1))
            y_poly = poly_model.final_poly_model.predict(x_poly)
            ax.plot(x_range, y_poly, color, label="Poly Fit Final")
            ax.legend()

    def visualize_decision_boundaries(self, poly_model=None):
        current_file_path = os.path.abspath(__file__)
        # Extract the directory part of the path
        current_dir = os.path.dirname(current_file_path)
        dir_pred = os.path.join(current_dir, "pixel_to_metric_predictors")
        validate_dir(dir_pred)
        
        fig, ax = plt.subplots(figsize=(18, 6))  # Create a single subplot
        all_features = self.features

        # Assuming you want to plot only the first combination for demonstration
        feature_combinations = list(itertools.combinations(range(len(all_features)), 2))
        if feature_combinations:  # Check if there are at least two features to combine
            i, j = feature_combinations[0]  # Get the first combination
            self.model.fit(self.X_train[[all_features[i], all_features[j]]], self.y_train)
            self.plot_decision_boundaries([i, j], ax, poly_model=poly_model if all_features[i] == 'MP' or all_features[j] == 'MP' else None)
        else:
            print("Not enough features to create combinations.")

        plt.tight_layout()
        file_path = os.path.join(dir_pred, "pixel_predictions.png")
        plt.savefig(file_path, dpi=600)
        plt.close(fig)

    def train(self):
        print("Training with features:", self.X_train.columns.tolist())
        self.model.fit(self.X_train, self.y_train)


    def evaluate(self):
        print("Evaluating with features:", self.X_test.columns.tolist())
        if self.X_test.shape[1] != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features, but got {self.X_test.shape[1]} features.")
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")

    def save_model(self, filename='rf_classifier.joblib'):
        current_file_path = os.path.abspath(__file__)

        # Extract the directory part of the path
        current_dir = os.path.dirname(current_file_path)
        dir_pred = os.path.join(current_dir, "pixel_to_metric_predictors")
        validate_dir(dir_pred)

        dump(self.model, os.path.join(dir_pred,filename))

    def load_model(self, filename='rf_classifier.joblib'):
        current_file_path = os.path.abspath(__file__)
        # Extract the directory part of the path
        current_dir = os.path.dirname(current_file_path)    
        dir_pred = os.path.join(current_dir, "pixel_to_metric_predictors")

        self.model = load(os.path.join(dir_pred,filename))

    def predict(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        new_data['MP'] = (new_data['image_height'] * new_data['image_width']) / 1000000
        new_features = new_data[self.features].fillna(0)

        # Check if the new data has the correct number of features
        if new_features.shape[1] != len(self.features):
            raise ValueError(f"New data must contain exactly {len(self.features)} features.")

        new_data['correct_conversion'] = self.model.predict(new_features)
        return new_data
    

class PolynomialModel:
    def __init__(self, data_path=None, features=['MP'], poly_degree=2):
        self.data_path = data_path
        self.features = features  # This should only include the 'MP' feature if that's what's used for the poly model
        self.poly_degree = poly_degree
        self.final_poly_model = None
        self.poly_transformer = PolynomialFeatures(degree=self.poly_degree, include_bias=False)

    def prepare_data(self):
        self.data['MP'] = (self.data['image_height'] * self.data['image_width']) / 1000000
        self.X = self.data[self.features]
        self.y = self.data['conversion_mean']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42)
        self.X_train.fillna(0, inplace=True)
        self.y_train.fillna(0, inplace=True)

    def train_polynomial_regressor(self):
        self.data = pd.read_csv(self.data_path)
        self.poly_model = None
        self.prepare_data()
        
        # Initial polynomial fitting
        self.poly_transformer.fit(self.X_train)  # Fit the PolynomialFeatures transformer
        X_train_poly = self.poly_transformer.transform(self.X_train)
        self.initial_poly_model = LinearRegression()
        self.initial_poly_model.fit(X_train_poly, self.y_train)
        
        # Predict and determine residuals
        y_pred = self.initial_poly_model.predict(X_train_poly)
        residuals = np.abs(self.y_train - y_pred)
        
        # Filter based on 1 standard deviation from residuals
        std_dev = np.std(residuals)
        within_one_std = residuals <= (std_dev * 0.5)
        
        # Final fitting only on filtered data
        X_train_poly_filtered = X_train_poly[within_one_std]
        y_train_filtered = self.y_train[within_one_std]
        self.final_poly_model = LinearRegression()
        self.final_poly_model.fit(X_train_poly_filtered, y_train_filtered)

    # def predict_with_polynomial(self, new_data_path):
    #     new_data = pd.read_csv(new_data_path)
    #     new_data['MP'] = (new_data['image_height'] * new_data['image_width']) / 1000000
    #     X_new = new_data[self.features].fillna(0)

    #     poly = PolynomialFeatures(degree=self.poly_degree)
    #     X_new_poly = poly.fit_transform(X_new)
    #     predictions = self.final_poly_model.predict(X_new_poly)

    #     new_data['predicted_conversion_mean'] = predictions
    #     return new_data
    def predict_with_polynomial(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        new_data['MP'] = (new_data['image_height'] * new_data['image_width']) / 1000000
        X_new = new_data[self.features].fillna(0)

        # Reuse the fitted transformer instead of creating a new one
        X_new_poly = self.poly_transformer.transform(X_new)  # Use transform, not fit_transform
        predictions = self.final_poly_model.predict(X_new_poly)

        new_data['predicted_conversion_mean'] = predictions
        return new_data

    
    def predict_with_polynomial_single(self, mp_value):
        if self.poly_transformer is None or self.final_poly_model is None:
            raise Exception("Model and transformer must be loaded or trained before prediction.")
        
        # Create a DataFrame with the same column names used in training
        mp_df = pd.DataFrame({'MP': [mp_value]})  # 'MP' is the feature name
        
        mp_transformed = self.poly_transformer.transform(mp_df)
        prediction = self.final_poly_model.predict(mp_transformed)
        return prediction[0]

    def save_polynomial_model(self, filename='poly_regressor_v1_5_2.joblib'):
        if self.final_poly_model:
            current_file_path = os.path.abspath(__file__)

            # Extract the directory part of the path
            current_dir = os.path.dirname(current_file_path)
            dir_pred = os.path.join(current_dir, "pixel_to_metric_predictors")
            validate_dir(dir_pred)

            dump((self.poly_transformer, self.final_poly_model), os.path.join(dir_pred,filename))
        else:
            print("Polynomial regressor is not trained.")

    def load_polynomial_model(self, filename='poly_regressor_v1_5_2.joblib'):
        current_file_path = os.path.abspath(__file__)
        # Extract the directory part of the path
        current_dir = os.path.dirname(current_file_path)    
        dir_pred = os.path.join(current_dir, "pixel_to_metric_predictors")

        self.poly_transformer, self.final_poly_model = load(os.path.join(dir_pred,filename))

    def export_predictions(self, predictions_df, save_path):
        predictions_df.to_csv(save_path, index=False)
        print(f"Predictions exported to {save_path}")

if __name__ == '__main__':
    features = ['MP', 'conversion_mean']

    rf_model = RandomForestModel('D:/D_Desktop/messy/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily_PRED.csv', features)
    rf_model.train()
    # rf_model.visualize_decision_boundaries()
    rf_model.evaluate()
    rf_model.save_model()
    rf_model.load_model()
    new_predictions = rf_model.predict('D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS.csv')

    features_poly = ['MP']
    poly_model = PolynomialModel('D:/D_Desktop/messy/LM2_random_forest_test_GBIF_BroadSample_3SppPerFamily_PRED.csv', features_poly)
    poly_model.train_polynomial_regressor()
    poly_model.save_polynomial_model()
    new_predictions_poly = poly_model.predict_with_polynomial('D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS.csv')
    poly_model.export_predictions(new_predictions_poly, 'D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS_POLYPREDICITONS.csv')

    # Visualize with polynomial regression in decision boundary plots
    # rf_model.visualize_decision_boundaries(poly_model=poly_model)


    poly_model = PolynomialModel()
    poly_model.load_polynomial_model('poly_regressor_v1_5_2.joblib')
    mp_value = 1.4  # Example MP value
    prediction = poly_model.predict_with_polynomial_single(mp_value)
    print(f'Predicted conversion mean for MP={mp_value}: {prediction}')

