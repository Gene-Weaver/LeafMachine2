import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np
import ast

def visualize_measurements(filename):
    size_pt = 1
    size_avg = 120

    # read the csv file
    df = pd.read_csv(filename)

    # group the individuals based on the filename column
    groups = df.groupby('wholename')

    # create subplots for all features
    fig, axes = plt.subplots(3, 3, figsize=(40, 40))

    # create a list of dummy lines for the legend
    lines = []
    num_colors = 16
    colors_all = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()  # get the first 20 colors from the 'tab20' color map
    colors_all.append([0.5, 0.5, 0.5, 1])  # add a custom color (gray) as the 21st color

    names = [
        "Ericaceae_Rhododendron_maximum",
        "Ericaceae_Vaccinium_arboreum",
        "Fagaceae_Castanea_dentata",
        "Fagaceae_Quercus_bicolor",
        "Fagaceae_Quercus_rubra",
        "Lauraceae_Lindera_benzoin",
        "Lauraceae_Sassafras_albidum",
        "Moraceae_Ficus_citrifolia",
        "Moraceae_Morus_alba",
        "Moraceae_Morus_rubra",
        "Ginkgoaceae_Ginkgo_biloba",
        "Sapindaceae_Acer_grandidentatum",
        "Plantaginaceae_Penstemon_eatonii",
        "Brassicaceae_Brassica_rapa",
        "Asteraceae_Helianthella_uniflora",
        "Betulaceae_Betula_papyrifera",
    ]

    colors = plt.cm.tab20(range(len(names)))

    name_color_dict = dict(zip(names, colors))

    # for name, color in name_color_dict.items():
    #     print(f"{name}: {color}")

    for i, (filename, group) in enumerate(groups):
        # color = colors[i % num_colors]
        color = name_color_dict[filename]
        line = Line2D([0], [0], color=color, lw=2)
        lines.append(line)

    # iterate through the groups and plot the scatter for each feature
    for i, (filename, group) in enumerate(tqdm(groups)):
        leaf_rows = group[group['annotation_name'].str.split('_').apply(lambda x: 'leaf' in x)]
        color = name_color_dict[filename]
        # Calculate the average of each column for filtered rows
        avg_row = leaf_rows.median(numeric_only=True)
        # iterate through the rows and process only if 'leaf' in row['annotation_name'].split('_')
        for ind, (row_index, row) in tqdm(enumerate(group.iterrows()), total=len(group), desc=f"Processing rows in {filename}"):
            if ind < 500:
                if 'leaf' in row['annotation_name'].split('_'):
                    # get the color for the group
                    
                    # plot the scatter for each feature
                    axes[0, 0].scatter(row['bbox_min_long_side'], row['bbox_min_short_side'], color=color, s=size_pt)
                    axes[0, 1].scatter(row['area'], row['perimeter'], color=color, s=size_pt)
                    cent = ast.literal_eval(row['centroid'])
                    axes[0, 2].scatter(cent[0], cent[1], color=color, s=size_pt)
                    axes[1, 0].scatter(row['convex_hull'], row['convexity'], color=color, s=size_pt)
                    axes[1, 1].scatter(row['convex_hull'], row['concavity'], color=color, s=size_pt)
                    axes[1, 2].scatter(row['convex_hull'], row['circularity'], color=color, s=size_pt)
                    axes[2, 0].scatter(row['convexity'], row['concavity'], color=color, s=size_pt)
                    axes[2, 1].scatter(row['convexity'], row['circularity'], color=color, s=size_pt)
                    axes[2, 2].scatter(row['concavity'], row['circularity'], color=color, s=size_pt)

        # Plot the average points with a black border
        axes[0, 0].scatter(avg_row['bbox_min_long_side'], avg_row['bbox_min_short_side'], color=color, s=size_avg, edgecolors='black')
        axes[0, 1].scatter(avg_row['area'], avg_row['perimeter'], color=color, s=size_avg, edgecolors='black')
        # axes[0, 2].scatter(avg_row['centroid'][0], avg_row['centroid'][1], color=color, s=size_avg, edgecolors='black')
        axes[1, 0].scatter(avg_row['convex_hull'], avg_row['convexity'], color=color, s=size_avg, edgecolors='black')
        axes[1, 1].scatter(avg_row['convex_hull'], avg_row['concavity'], color=color, s=size_avg, edgecolors='black')
        axes[1, 2].scatter(avg_row['convex_hull'], avg_row['circularity'], color=color, s=size_avg, edgecolors='black')
        axes[2, 0].scatter(avg_row['convexity'], avg_row['concavity'], color=color, s=size_avg, edgecolors='black')
        axes[2, 1].scatter(avg_row['convexity'], avg_row['circularity'], color=color, s=size_avg, edgecolors='black')
        axes[2, 2].scatter(avg_row['concavity'], avg_row['circularity'], color=color, s=size_avg, edgecolors='black')

    # add labels and titles for the subplots
    axes[0, 0].set(xlabel='bbox_min_long_side', ylabel='bbox_min_short_side', title='Scatter plot for bbox_min_long_side and bbox_min_short_side')
    axes[0, 1].set(xlabel='area', ylabel='perimeter', title='Scatter plot for area and perimeter')
    axes[0, 2].set(xlabel='centroid_x', ylabel='centroid_y', title='Scatter plot for centroid')
    axes[1, 0].set(xlabel='convex_hull', ylabel='convexity', title='Scatter plot for convex_hull and convexity')
    axes[1, 1].set(xlabel='convex_hull', ylabel='concavity', title='Scatter plot for convex_hull and concavity')
    axes[1, 2].set(xlabel='convex_hull', ylabel='circularity', title='Scatter plot for convex_hull and circularity')
    axes[2, 0].set(xlabel='convexity', ylabel='concavity', title='Scatter plot for convexity and concavity')
    axes[2, 1].set(xlabel='convexity', ylabel='circularity', title='Scatter plot for convexity and circularity')
    axes[2, 2].set(xlabel='concavity', ylabel='circularity',title='Scatter plot for concavity and circularity')


    # adjust the space between the subplots
    fig.subplots_adjust(hspace=0.08, wspace=0.08, right=0.9, left=0.04, bottom=0.04, top=0.96)
    fig.legend(lines, groups.groups.keys(), loc='center left', bbox_to_anchor=(.91, 0.5))

    # show the plot
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\measurements_plot_full_run.png', dpi=300)


if __name__ == '__main__':
    filename = 'D:\Dropbox\LM2_Env\Image_Datasets\TEST_LM2\Explore\merged_data.csv'
    visualize_measurements(filename)