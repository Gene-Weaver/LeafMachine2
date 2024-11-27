import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import itertools
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
# new_data_file_path = 'D:/T_Downloads/LM2_Populus_subset_MEASUREMENTS_PRED_MEAN_POLY_best.csv'
# new_data_file_path = 'D:/T_Downloads/LM2_Quercus_NA-500_MEASUREMENTS_PRED_MEAN_POLY_best.csv'
# new_data_file_path = 'D:/T_Downloads/LM2_Quercus_NA-500_AND_Populus_MEASUREMENTS_PRED_MEAN_POLY_best_el.csv'
new_data_file_path = 'G:/Thais/LM2/Data/Measurements/LM2_MEASUREMENTS_CLEAN.csv'

# pdf_path = 'D:/T_Downloads/violin_plots_per_class_Quercus_AND_Populus_Genus_el.pdf'
# new_data_file_path_LANDMARK = 'D:/T_Downloads/LM2_Quercus_NA_AND_Populus_LANDMARKS_el.csv'
pdf_path = 'G:/Thais/LM2/Data/Measurements/violin_plots_per_class.pdf'
new_data_file_path_LANDMARK = 'G:/Thais/LM2/Data/Landmarks/LM2_LANDMARKS.csv'
path_circularity_convexity = 'G:/Thais/LM2/Data/Measurements/circularity_convexity.pdf'

new_data = pd.read_csv(new_data_file_path)

# Extract the relevant part of the filename and create a new column for it
# new_data['class'] = new_data['filename'].apply(lambda x: '_'.join(x.split('_')[2:5]))
# new_data['class'] = new_data['filename'].apply(lambda x: '_'.join(x.split('_')[2:4])) # For GBIF
new_data['class'] = new_data['fullname'] # For Thais
print(new_data['class'])
# Count the number of rows per class
class_counts = new_data['class'].value_counts()

MIN_N_LEAVES = 10

classes_to_keep = class_counts[class_counts >= MIN_N_LEAVES].index
new_data = new_data[new_data['class'].isin(classes_to_keep)]
print(class_counts)
# Get unique classes for color mapping
classes = new_data['class'].unique()
colors = plt.cm.jet(np.linspace(0, 1, len(classes)))
color_map = dict(zip(classes, colors))




# Select the columns for PCA and ANOVA
columns = ['area', 'perimeter', 'convex_hull', 'convexity', 'circularity', 'aspect_ratio']
# columns = ['area', 'perimeter', 'convex_hull', 'convexity', 'concavity', 'circularity', 'aspect_ratio']


# Prepare violin plot data
# sorted_classes = class_counts[class_counts >= MIN_N_LEAVES].sort_values().index
sorted_classes = sorted(class_counts[class_counts >= MIN_N_LEAVES].index)


# pdf_path = 'D:/T_Downloads/violin_plots_per_class_Populus_log.pdf'
pdf_pages = PdfPages(pdf_path)

# Set the Seaborn style and font scale before the loop to apply it to all plots
sns.set(font_scale=1.5)

for col in columns:
    # Calculate the figure height based on the number of classes for better fit
    fig_height = len(sorted_classes) * 0.5  # Adjust the multiplier as needed for your data
    fig, ax = plt.subplots(figsize=(15, max(fig_height, 20)))  # Ensure a minimum height for the figure

    sns.violinplot(x=col, y='class', data=new_data, order=sorted_classes, cut=0)
    ax.set_title(col)

    # Adjust the layout to make room for class labels and title
    plt.tight_layout()
    plt.subplots_adjust(left=0.3, top=0.95)

    # Save the figure to the PDF
    pdf_pages.savefig(fig)
    plt.close(fig)




############################## ANGLES
# new_data_file_path_LANDMARK = 'D:/T_Downloads/LM2_Populus_subset_LANDMARKS_CFapplied_PRED_MEAN_POLY_best.csv'
new_data_LANDMARK = pd.read_csv(new_data_file_path_LANDMARK)
columns_LANDMARK = ['apex_angle_degrees', 'base_angle_degrees']
# new_data_LANDMARK['class'] = new_data_LANDMARK['filename'].apply(lambda x: '_'.join(x.split('_')[2:5]))
new_data_LANDMARK['class'] = new_data_LANDMARK['filename'].apply(lambda x: '_'.join(x.split('_')[2:4]))
print(new_data_LANDMARK['class'])
# Count the number of rows per class
class_counts_LANDMARK = new_data_LANDMARK['class'].value_counts()

classes_to_keep_LANDMARK = class_counts_LANDMARK[class_counts_LANDMARK >= MIN_N_LEAVES].index
new_data_LANDMARK = new_data_LANDMARK[new_data_LANDMARK['class'].isin(classes_to_keep)]
# sorted_classes_LANDMARK = class_counts_LANDMARK[class_counts_LANDMARK >= MIN_N_LEAVES].sort_values().index
sorted_classes_LANDMARK = sorted(class_counts_LANDMARK[class_counts_LANDMARK >= MIN_N_LEAVES].index)


for col in columns_LANDMARK:
    # Calculate the figure height based on the number of classes for better fit
    fig_height = len(sorted_classes_LANDMARK) * 0.5  # Adjust the multiplier as needed for your data
    fig, ax = plt.subplots(figsize=(15, max(fig_height, 20)))  # Ensure a minimum height for the figure

    sns.violinplot(x=col, y='class', data=new_data_LANDMARK, order=sorted_classes_LANDMARK, cut=0)
    ax.set_title(col)

    # Adjust the layout to make room for class labels and title
    plt.tight_layout()
    plt.subplots_adjust(left=0.3, top=0.95)

    # Save the figure to the PDF
    pdf_pages.savefig(fig)
    plt.close(fig)
#################################### ANGLES



######################### PCA
# Load the data
# new_data_file_path = 'D:/T_Downloads/LM2_Populus_subset_MEASUREMENTS_PRED_MEAN_POLY_best.csv'
new_data = pd.read_csv(new_data_file_path)

# Log-transform 'area', 'perimeter', and 'convex_hull' columns
for col in ['area', 'perimeter', 'convex_hull']:
    new_data[col] = new_data[col].apply(lambda x: np.log(x) if x > 0 else None)

# Extract the relevant part of the filename and create a new class column for the first dataset
# new_data['class'] = new_data['filename'].apply(lambda x: '_'.join(x.split('_')[2:5]))
new_data['class'] = new_data['filename'].apply(lambda x: '_'.join(x.split('_')[2:4]))

# Load the landmark data
# new_data_file_path_LANDMARK = 'D:/T_Downloads/LM2_Populus_subset_LANDMARKS_CFapplied_PRED_MEAN_POLY_best.csv'
new_data_LANDMARK = pd.read_csv(new_data_file_path_LANDMARK)

# Merge the two datasets on the 'component_name' column
combined_data = pd.merge(new_data, new_data_LANDMARK, on='component_name', how='inner', suffixes=('_shape', '_landmark'))

# Filter classes with at least MIN_N_LEAVES from the combined dataset
class_counts = combined_data['class'].value_counts()



classes_to_keep = class_counts[class_counts >= MIN_N_LEAVES].index###################################################
combined_data = combined_data[combined_data['class'].isin(classes_to_keep_LANDMARK)]




# Now you can perform PCA and ANOVA on the combined dataset
# Select the columns for PCA and ANOVA
columns_shape = ['area', 'perimeter', 'convex_hull', 'convexity', 'circularity', 'aspect_ratio']
# columns_shape = ['convexity',  'circularity', 'aspect_ratio']
columns_landmark = ['apex_angle_degrees', 'base_angle_degrees']

# Assuming columns_shape are from new_data and columns_landmark are from new_data_LANDMARK
pca_data = combined_data[columns_shape + columns_landmark]

# Handle missing data
pca_data.dropna(inplace=True)

# Standardize the data before PCA
scaler = StandardScaler()
pca_data_std = scaler.fit_transform(pca_data)

# Perform PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(pca_data_std)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
finalDf = pd.concat([principalDf, combined_data[['class']]], axis=1)
finalDf = finalDf.dropna(subset=['class'])

# ANOVA test
anova_results = {col: stats.f_oneway(*[group[col].values for name, group in combined_data.groupby('class')])
                 for col in columns_shape + columns_landmark}
print(anova_results)
print(finalDf)


# Print explained variance
print("Explained variance by component:", pca.explained_variance_ratio_)

# Print PCA loadings
print("\nPCA Loadings:")
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=columns_shape + columns_landmark)
print(loadings)

# Perform and print ANOVA results
print("\nANOVA Results:")
for col in columns_shape + columns_landmark:
    # Perform ANOVA and print F-statistic and p-value
    groups = [group[col].dropna().values for name, group in combined_data.groupby('class')]
    if len(groups) > 1:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"{col}: F-statistic = {f_stat}, p-value = {p_value}")




# Plot PCA scatterplot with legend outside the plot area
classes = finalDf['class'].unique()
color_map = {class_: color for class_, color in zip(classes, plt.cm.jet(np.linspace(0, 1, len(classes))))}

class_means = finalDf.groupby('class').mean()
class_stds = finalDf.groupby('class').std()
print(class_means)
print(class_stds)
# Define line styles for different groups of ellipses
line_styles = ['-', '--', ':']
line_cycle = itertools.cycle(line_styles)  # Create a cycle iterator for line styles

# Plot PCA scatterplot
fig, ax = plt.subplots(figsize=(12, 8))
# Set x and y axis limits
# ax.set_xlim(-4, 6)
# ax.set_ylim(-4, 3)

# Track the line style for the legend
line_style_for_legend = {}

# Plot the points with reduced opacity
for classname, group in finalDf.groupby('class'):
    ax.scatter(group['PC1'], group['PC2'], alpha=0.0, label=classname, color=color_map[classname])

# Plot an ellipse for each class with different line styles
for i, (classname, group) in enumerate(finalDf.groupby('class')):
    mean = class_means.loc[classname]
    std = class_stds.loc[classname]
    if i % 10 == 0:
        line_style = next(line_cycle)
    line_style_for_legend[classname] = line_style
    ellipse = Ellipse(xy=(mean['PC1'], mean['PC2']), width=std['PC1']*2, height=std['PC2']*2,
                      edgecolor=color_map[classname], linestyle=line_style, fc='None', lw=1)
    ax.add_patch(ellipse)

# Create legend handles that match the ellipses
legend_handles = [Line2D([0], [0], color=color_map[classname], linestyle=line_style_for_legend[classname],
                         label=classname, lw=2) for classname in classes]


# Set title and labels
ax.set_title('PCA: PC1 vs PC2 with Class Ellipses')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# Add the custom legend to the plot
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='xx-small')


# Adjust the layout
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect to prevent cutting off the legend
# Save PCA plot to the PDF
pdf_pages.savefig(fig)
plt.close(fig)
######################### PCA


######################### QDA
# # Separate features and class labels
# X = combined_data[columns_shape + columns_landmark]  # Features
# y = combined_data['class']  # Class labels

# # Handling missing data
# X.dropna(inplace=True)
# y = y[X.index]  # Ensuring the labels correspond to the remaining data after dropping NaNs

# # Standardize the features
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)

# # Quadratic Discriminant Analysis
# qda = QuadraticDiscriminantAnalysis()
# qda.fit(X_std, y)

# # Predictions
# y_pred = qda.predict(X_std)

# # Evaluation
# conf_matrix = confusion_matrix(y, y_pred)
# print("Confusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", classification_report(y, y_pred))

# # Separate features and class labels
# X = combined_data[columns + columns_LANDMARK]  # Features
# y = combined_data['class']  # Class labels
# X = X.dropna()
# y = y[X.index]

# # Encoding the class labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Dimensionality Reduction using PCA
# pca = PCA(n_components=2)
# X_r = pca.fit_transform(X)

# # Perform QDA
# qda = QuadraticDiscriminantAnalysis()
# qda.fit(X_r, y_encoded)

# # Visualization
# fig, ax = plt.subplots(figsize=(12, 8))

# # Define color map and line styles for different classes
# color_map = {class_: color for class_, color in zip(label_encoder.classes_, plt.cm.jet(np.linspace(0, 1, len(label_encoder.classes_))))}
# line_styles = ['-', '--', ':']
# line_cycle = itertools.cycle(line_styles)
# line_style_for_legend = {}

# for i, class_name in enumerate(label_encoder.classes_):
#     class_index = label_encoder.transform([class_name])[0]
#     class_data = X_r[y_encoded == class_index, :]
    
#     mean = np.mean(class_data, axis=0)
#     std = np.std(class_data, axis=0)

#     # Update line style every 10 classes
#     if i % 10 == 0:
#         line_style = next(line_cycle)
#     line_style_for_legend[class_name] = line_style

#     # Create and add ellipse to plot
#     ellipse = Ellipse(xy=mean, width=std[0]*2, height=std[1]*2, edgecolor=color_map[class_name], linestyle=line_style, fc='None', lw=1)
#     ax.add_patch(ellipse)
#     ax.scatter(class_data[:, 0], class_data[:, 1], label=class_name, color=color_map[class_name], alpha=0.0)

# # Add a legend for the classes
# legend_handles = [Line2D([0], [0], color=color_map[class_name], linestyle=line_style_for_legend[class_name],
#                          label=class_name, lw=2) for class_name in label_encoder.classes_]

# # Set title and labels
# ax.set_xlabel('PCA Component 1')
# ax.set_ylabel('PCA Component 2')
# ax.set_title('Class Separation using QDA and PCA')

# # Add the custom legend to the plot
# ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='xx-small')
# plt.tight_layout(rect=[0, 0, 0.85, 1]) 
# pdf_pages.savefig(fig)
# plt.close(fig)
######################### QDA



######################## MANOVA
# manova_formula = 'convexity + circularity + aspect_ratio + apex_angle_degrees + base_angle_degrees ~ class'

# # Perform MANOVA
# manova = MANOVA.from_formula(manova_formula, data=combined_data)
######################## MANOVA


######################## LDA
# Preparing the data
X = combined_data[columns + columns_LANDMARK]
y = combined_data['class']
X = X.dropna()
y = y[X.index] 

# Encoding the class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform LDA
lda = LinearDiscriminantAnalysis(n_components=2)  # Assuming you want 2 components CCA
X_r = lda.fit(X, y_encoded).transform(X)

# Print the explained variance ratio
print('Explained variance ratio of the 2 LDA components:', lda.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_xlim(-3, 8)
# ax.set_ylim(-6, 4)
# Plot an ellipse for each class with different line styles
for i, class_name in enumerate(label_encoder.classes_):
    class_index = label_encoder.transform([class_name])[0]
    class_data = X_r[y_encoded == class_index, :]
    
    mean = np.mean(class_data, axis=0)
    std = np.std(class_data, axis=0)

    class_index = label_encoder.transform([class_name])[0]
    ax.scatter(X_r[y_encoded == class_index, 0], X_r[y_encoded == class_index, 1], 
               label=class_name, color=color_map[class_name], alpha=0.0)

    # Update line style every 10 classes
    if i % 10 == 0:
        line_style = next(line_cycle)
    line_style_for_legend[class_name] = line_style

    # Create and add ellipse to plot
    ellipse = Ellipse(xy=mean, width=std[0]*2, height=std[1]*2, edgecolor=color_map[class_name], linestyle=line_style, fc='None', lw=1)
    ax.add_patch(ellipse)

# Add a legend for the classes
# Create legend handles that match the ellipses
legend_handles = [Line2D([0], [0], color=color_map[class_name], linestyle=line_style_for_legend[class_name],
                         label=class_name, lw=2) for class_name in label_encoder.classes_]

# Set title and labels
ax.set_xlabel('LDA Component 1')
ax.set_ylabel('LDA Component 2')
ax.set_title('Class Separation using LDA')

# Add the custom legend to the plot
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='xx-small')
plt.tight_layout(rect=[0, 0, 0.85, 1]) 

# Access the coefficients
lda_coefs = lda.coef_

# Map coefficients to original features
feature_names = columns + columns_LANDMARK  # List of original feature names
for i, component in enumerate(lda_coefs, start=1):
    print(f"LDA Component {i}:")
    for feature, coef in zip(feature_names, component):
        print(f"    {feature}: {coef}")
# Save PCA plot to the PDF
pdf_pages.savefig(fig)
plt.close(fig)

for class_name in label_encoder.classes_:
    class_index = label_encoder.transform([class_name])[0]
    class_data = X_r[y_encoded == class_index, 0]  # Only the first component
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.hist(class_data, bins=30, color=color_map[class_name], alpha=0.7)
    plt.title(f'Histogram of LDA1 for {class_name}')
    plt.xlabel('LDA Component 1')
    plt.ylabel('Frequency')
    pdf_pages.savefig(fig)
    plt.close(fig)

######################## LDA












# Close the PDF file
pdf_pages.close()












for cls in classes:
    # Filter data for each class
    class_data = new_data[new_data['class'] == cls]

    # Calculate the mean for aspect_ratio and convexity for the current class
    mean_aspect_ratio = class_data['circularity'].mean()
    mean_convexity = class_data['convexity'].mean()

    # Calculate the standard deviation for aspect_ratio and convexity for the current class
    std_aspect_ratio = class_data['circularity'].std()
    std_convexity = class_data['convexity'].std()

    # Plot the mean values as points
    plt.scatter(mean_aspect_ratio, mean_convexity, 
                color=color_map[cls], label=cls, s=100)  # s is the size of the point

    # Plot an ellipse around each mean point to show variance
    ellipse = plt.matplotlib.patches.Ellipse((mean_aspect_ratio, mean_convexity), 
                                             2*std_aspect_ratio, 2*std_convexity, 
                                             color=color_map[cls], fill=False, linewidth=2, alpha=0.5)
    plt.gca().add_patch(ellipse)


plt.xlabel('Circularity')
plt.ylabel('Convexity')
plt.title('Average Morphospace by Class')
plt.grid(True)
# plt.legend()

# Save the plot to a PDF file 
plt.savefig(path_circularity_convexity)

# Close the plot
plt.close()



















# Plotting
plt.figure(figsize=(10, 8))

for cls in classes:
    # Filter data for each class
    class_data = new_data[new_data['class'] == cls]

    # Plot data points
    plt.scatter(class_data['aspect_ratio'], class_data['circularity'], 
                color=color_map[cls], label=cls)

    # Calculate the mean for aspect_ratio and circularity for the current class
    mean_aspect_ratio = class_data['aspect_ratio'].mean()
    mean_circularity = class_data['circularity'].mean()

    # Calculate the standard deviation for aspect_ratio and circularity for the current class
    std_aspect_ratio = class_data['aspect_ratio'].std()
    std_circularity = class_data['circularity'].std()

    # Determine the radius for the circle (e.g., using the average of the standard deviations)
    radius = (std_aspect_ratio + std_circularity) / 2

    # Plot a large circle around the morphospace of each class
    ellipse = plt.Circle((mean_aspect_ratio, mean_circularity), radius, 
                         color=color_map[cls], fill=False, linewidth=2, alpha=0.5)
    plt.gca().add_artist(ellipse)

plt.xlabel('Aspect Ratio')
plt.ylabel('Circularity')
plt.title('Morphospace by Class')
# plt.legend()
plt.grid(True)
# plt.show()

# Save the plot to a PDF file
plt.savefig('D:/T_Downloads/populus_aspect_ratio_circularity.pdf')

# Close the plot
plt.close()


# Plotting
plt.figure(figsize=(10, 8))

for cls in classes:
    # Filter data for each class
    class_data = new_data[new_data['class'] == cls]

    # Plot data points
    plt.scatter(class_data['aspect_ratio'], class_data['convexity'], 
                color=color_map[cls], label=cls)

    # Calculate the mean for aspect_ratio and convexity for the current class
    mean_aspect_ratio = class_data['aspect_ratio'].mean()
    mean_convexity = class_data['convexity'].mean()

    # Calculate the standard deviation for aspect_ratio and convexity for the current class
    std_aspect_ratio = class_data['aspect_ratio'].std()
    std_convexity = class_data['convexity'].std()

    # Determine the radius for the circle (e.g., using the average of the standard deviations)
    radius = (std_aspect_ratio + std_convexity) / 2

    # Plot a large circle around the morphospace of each class
    ellipse = plt.Circle((mean_aspect_ratio, mean_convexity), radius, 
                         color=color_map[cls], fill=False, linewidth=2, alpha=0.5)
    plt.gca().add_artist(ellipse)

plt.xlabel('Aspect Ratio')
plt.ylabel('convexity')
plt.title('Morphospace by Class')
# plt.legend()
plt.grid(True)
# plt.show()

# Save the plot to a PDF file
plt.savefig('D:/T_Downloads/populus_aspect_ratio_convexity.pdf')

# Close the plot
plt.close()