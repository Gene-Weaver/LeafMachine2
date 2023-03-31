import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_measurements(filename):
    size_pt = 20


    # read the csv file
    df = pd.read_csv(filename)

    # group the individuals based on the filename column
    groups = df.groupby('group')

    # create subplots for all features
    fig, axes = plt.subplots(3, 3, figsize=(40, 40))

    # create a list of dummy lines for the legend
    lines = []
    for i, (filename, group) in enumerate(groups):
        color = plt.cm.Set1(i/len(groups))
        line = Line2D([0], [0], color=color, lw=2)
        lines.append(line)

    # iterate through the groups and plot the scatter for each feature
    for i, (filename, group) in enumerate(groups):
        # iterate through the rows and process only if 'leaf' in row['annotation_name'].split('_')
        for _, row in group.iterrows():
            if 'leaf' in row['annotation_name'].split('_'):
                # get the color for the group
                color = plt.cm.Set1(i/len(groups))
                # plot the scatter for each feature
                axes[0, 0].scatter(row['bbox_min_long_side'], row['bbox_min_short_side'], color=color, s=size_pt)
                axes[0, 1].scatter(row['area'], row['perimeter'], color=color, s=size_pt)
                axes[0, 2].scatter(row['centroid'][0], row['centroid'][1], color=color, s=size_pt)
                axes[1, 0].scatter(row['convex_hull'], row['convexity'], color=color, s=size_pt)
                axes[1, 1].scatter(row['convex_hull'], row['concavity'], color=color, s=size_pt)
                axes[1, 2].scatter(row['convex_hull'], row['circularity'], color=color, s=size_pt)
                axes[2, 0].scatter(row['convexity'], row['concavity'], color=color, s=size_pt)
                axes[2, 1].scatter(row['convexity'], row['circularity'], color=color, s=size_pt)
                axes[2, 2].scatter(row['concavity'], row['circularity'], color=color, s=size_pt)

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
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.legend(lines, groups.groups.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    # show the plot
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\measurements_plot.png', dpi=300)


if __name__ == '__main__':
    filename = 'D:\Dropbox\LM2_Env\Image_Datasets\TEST_LM2\Explore_single_whole_50p\Data\Measurements\Explore_single_whole_50p.csv'
    visualize_measurements(filename)