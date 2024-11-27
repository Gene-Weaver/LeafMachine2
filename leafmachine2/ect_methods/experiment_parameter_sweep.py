from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS 
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


'''
affinity propagation clustering algorithm
https://scikit-learn.org/1.5/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py

'''
currentdir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(currentdir)
print(currentdir)
from leafmachine2.ect_methods.utils_ect import torch_distance_matrix, plot_confidence_ellipse, plot_confidence_ellipse_mean, calculate_ellipse_overlap, plot_bounding_ellipse
from leafmachine2.ect_methods.leaf_ect import LeafECT

def run_experiment_parameter_sweep(cleaned_df, outline_path, dir_experiments, bounding_fcn='convex_hull', plot_mean_shape=True, use_DP=True):

    pdf_filename = os.path.join(dir_experiments, 'experiment_parameter_sweep_50_taxa_withDP.pdf')

    NUM_SAMPLES = 5  # Number of samples per species, -1 will do EVERYTHING

    # NUM_DIRS_VALS = [16, 20, 50, 100]
    # NUM_THRESH_VALS =[20, 50, 100, 200] 
    # NUM_PTS_VALS = [5,25,50,100,200,500,1000,100000] #100,000 is just a maximum to ensure that none are shortened
    # NUM_DIRS_VALS = [16, 20,]
    # NUM_THRESH_VALS =[5, 20, 25, 50,] 
    # NUM_PTS_VALS = [5, 100,] 

    # INCLUDED_TAXA = [
    #     "Quercus_alba",
    #     "Annona_montana",
    #     "Acer_rubrum",
    #     "Lindera_benzoin",
    # ]

    NUM_DIRS_VALS = [4, 8, 16, 32, 64]
    NUM_THRESH_VALS = [20, 50, 100, 200, 400] 
    if use_DP:
        NUM_PTS_VALS = ['DP'] 
    else:
        NUM_PTS_VALS = [50,500,1000]
        
    INCLUDED_TAXA = [
        "Trema_cannabina",
        "Ginkgo_biloba",
        "Rhododendron_maximum",
        "Quercus_alba",
        "Prunus_serotina",
        "Vaccinium_angustifolium",
        "Parthenocissus_quinquefolia",
        "Morus_alba",
        "Alnus_incana",
        "Annona_montana",
        "Quercus_macrocarpa",
        "Gymnocladus_dioicus",
        "Gleditsia_triacanthos",
        "Prunus_virginiana",
        "Ulmus_americana",
        "Vitis_riparia",
        "Byrsonima_intermedia",
        "Lonicera_maackii",
        "Viburnum_acerifolium",
        "Lecythis_pisonis",
        "Alnus_rubra",
        "Rhamnus_cathartica",
        "Acer_saccharum",
        "Acer_rubrum",
        "Notholithocarpus_densiflorus",
        "Quercus_velutina",
        "Couroupita_guianensis",
        "Tetrapterys_mucronata",
        "Celtis_occidentalis",
        "Dianthus_armeria",
        "Robinia_pseudoacacia",
        "Raphanus_raphanistrum",
        "Cephalanthus_occidentalis",
        "Nicotiana_tabacum",
        "Liquidambar_styraciflua",
        "Asimina_triloba",
        "Quercus_gambelii",
        "Lonicera_periclymenum",
        "Vaccinium_oxycoccos",
        "Rosa_multiflora",
        "Rubus_occidentalis",
        "Tiarella_cordifolia",
        "Brassica_rapa",
        "Lindera_benzoin",
        "Viburnum_opulus",
        "Populus_deltoides",
        "Hieracium_umbellatum",
        "Frangula_alnus",
        "Populus_tremuloides",
        "Gaultheria_procumbens",
        "Cannabis_sativa",
    ]

    # Create directory if not exists
    os.makedirs(dir_experiments, exist_ok=True)

    # Initialize the PDF file

    # Data storage for later plotting
    s_scores = []
    overlap_areas_data = []
    params_data = []

    with PdfPages(pdf_filename) as pdf:

        for max_pts in NUM_PTS_VALS:
            for num_thresh in NUM_THRESH_VALS:
                for num_dirs in NUM_DIRS_VALS:
                    print(f"Running experiment with num_dirs={num_dirs}, num_thresh={num_thresh}, max_pts={max_pts}")

                    '''
                    Create ECT matrices
                    '''
                    taxa_list = [taxa for taxa in cleaned_df['genus_species'].unique() if taxa in INCLUDED_TAXA]
                    ect_data, group_labels, shapes = [], [], []

                    for taxa in taxa_list:
                        species_df = cleaned_df[cleaned_df['genus_species'] == taxa]
                        if NUM_SAMPLES == -1:
                            component_names = species_df['component_name'].tolist()  # Use all contours instead of sampling
                        else:
                            component_names = species_df['component_name'].sample(NUM_SAMPLES, random_state=42).tolist()  # Randomly sample 10 contours per species
                        
                        for component_name in component_names:
                            leaf_ect = LeafECT(cleaned_df, outline_path, num_dirs=num_dirs, num_thresh=num_thresh, max_pts=max_pts)
                            ECT_matrix, points = leaf_ect.compute_ect_for_contour(component_name, max_pts)

                            if ECT_matrix is not None:
                                ect_data.append(ECT_matrix)
                                group_labels.append(taxa)
                                shapes.append(points)
                    
                    '''
                    Create Distance matrices and MDS
                    '''
                    # Check if data exists
                    if len(ect_data) == 0:
                        print(f"No valid ECT data for num_dirs={num_dirs}, num_thresh={num_thresh}, max_pts={max_pts}")
                        continue

                    ect_arr = np.array(ect_data)
                    flattened_ect = ect_arr.reshape(len(ect_data), num_dirs * num_thresh)
                    sorted_indices = np.argsort(group_labels)
                    sorted_group_labels = np.array(group_labels)[sorted_indices]
                    sorted_flattened_ect = flattened_ect[sorted_indices]
                    sorted_flattened_ect_gpu = torch.tensor(sorted_flattened_ect, dtype=torch.float32, device='cuda')

                    D_sorted_gpu = torch_distance_matrix(sorted_flattened_ect_gpu)
                    D = D_sorted_gpu.cpu().numpy()

                    # Step 3: Perform MDS
                    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
                    mds_scores = mds.fit_transform(D)

                    # Convert group labels to numeric values using LabelEncoder for silhouette score
                    label_encoder = LabelEncoder()
                    numeric_labels = label_encoder.fit_transform(sorted_group_labels)
                    
                    s_score = silhouette_score(mds_scores, numeric_labels)
                    print(f'Silhouette Score: {s_score}')
                    s_scores.append(s_score)

                    # Step 4: Generate Plots (Distance matrix and MDS plot)
                    fig, axs = plt.subplots(1, 2, figsize=(15, 8))  # Landscape format

                    # Plot Distance Matrix
                    axs[0].matshow(D, cmap='viridis')
                    axs[0].set_title("Distance Matrix")
                    axs[0].set_xlabel("Samples")
                    axs[0].set_ylabel("Samples")
                    
                    '''
                    Plots
                    '''
                    ax = axs[1]
                    shape_scale_factor = 0.5
                    scale_val = 6
                    cmap = plt.get_cmap('tab20')
                    ellipses = []


                    for i, genus in tqdm(enumerate(taxa_list), total=len(taxa_list), desc="Processing Genera"):
                        species_idx = [idx for idx, label in enumerate(group_labels) if label == genus]

                        if plot_mean_shape:
    
                            # Plot the confidence ellipse and get its center
                            # if bounding_fcn == 'confidence':
                            ellipse_center = plot_confidence_ellipse_mean(mds_scores, species_idx, ax, edgecolor=cmap(i % 20), facecolor='none', lw=0.5)
                            # elif bounding_fcn == 'convex_hull':
                                # ellipse_center = plot_convex_hull_mean(mds_scores, species_idx, ax, color=cmap(i % 20), linewidth=0.5)

                            # Find the shape closest to the center of the ellipse
                            genus_mds_points = mds_scores[species_idx]
                            distances_to_center = np.linalg.norm(genus_mds_points - ellipse_center, axis=1)
                            closest_idx_within_genus = species_idx[np.argmin(distances_to_center)]  # Index of the closest shape

                            # Get the contour points for the closest shape
                            points = shapes[closest_idx_within_genus]  # Get the closest shape's contour

                            # Center and scale the shape
                            points = points - np.mean(points, axis=0)  # Zero-center the shape
                            points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale to radius 1 and then scale_val
                            points *= shape_scale_factor  # Reduce shape size by 10%

                            # Translate the shape to its MDS coordinates
                            trans_sh = points + mds_scores[closest_idx_within_genus]

                            # Plot the translated shape, color-coded by genus
                            plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(i % 20), lw=0, alpha=0.6)

                        else:
                            for idx in species_idx:
                                points = shapes[idx]
                                points = points - np.mean(points, axis=0)
                                points = scale_val * points / max(np.linalg.norm(points, axis=1))
                                points *= shape_scale_factor
                                trans_sh = points + mds_scores[idx]

                                ax.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(i % 20), lw=0, alpha=0.6)

                        if bounding_fcn == 'confidence':
                            ellipse_params = plot_confidence_ellipse(mds_scores, species_idx, ax, edgecolor=cmap(i % 20), facecolor='none', lw=0.5)
                            ellipses.append(ellipse_params)
                        elif bounding_fcn == 'convex_hull':
                            hull_params  = plot_bounding_ellipse(mds_scores, species_idx, ax, edgecolor=cmap(i % 20), facecolor='none', lw=0.5)
                            ellipses.append(hull_params)
                            
                    # Calculate total overlap between ellipses
                    overlap = 0
                    for j in range(len(ellipses)):
                        for k in range(j + 1, len(ellipses)):
                            # if bounding_fcn == 'confidence':
                            overlap += calculate_ellipse_overlap(ellipses[j], ellipses[k])
                            # elif bounding_fcn == 'convex_hull':
                                # overlap += calculate_convex_hull_overlap(ellipses[j], ellipses[k])

                    # Store ellipse areas, overlap, and parameters for later plotting
                    overlap_areas_data.append(overlap)
                    params_data.append((max_pts, num_thresh, num_dirs))

                    ax.set_aspect("equal", adjustable='box')
                    ax.set_title("MDS Plot")
                    ax.set_xlabel("MDS Dimension 1")
                    ax.set_ylabel("MDS Dimension 2")

                    # Step 5: Save to the PDF
                    plt.suptitle(f'num_dirs={num_dirs}, num_thresh={num_thresh}, max_pts={max_pts}', fontsize=16)
                    pdf.savefig(fig)  # Save the current figure as a page in the PDF
                    plt.close(fig)

        # # After looping, plot ellipse areas and overlap areas against the _VALS parameters
        # # Unpack the parameters from params_data for plotting
        # max_pts_vals, num_thresh_vals, num_dirs_vals = zip(*params_data)

        # # Create DataFrame for easier plotting
        # df = pd.DataFrame({
        #     'num_dirs': num_dirs_vals,
        #     'num_thresh': num_thresh_vals,
        #     'max_pts': max_pts_vals,
        #     'overlap': overlap_areas_data
        # })

        # # Convert num_thresh and num_dirs to categorical for better plotting
        # df['num_thresh'] = df['num_thresh'].astype('category')
        # df['num_dirs'] = df['num_dirs'].astype('category')

        # # Create a FacetGrid for each max_pts value with a larger figure size to prevent squishing
        # g = sns.FacetGrid(df, col="max_pts", height=5, aspect=1.5)  # Increased aspect ratio

        # # Use a barplot to show overlap as a function of num_thresh and num_dirs
        # g.map(sns.barplot, "num_thresh", "overlap", "num_dirs", palette="viridis")

        # # Adjust titles and labels
        # g.add_legend(title="num_dirs")
        # g.set_axis_labels("num_thresh", "Overlap Area")
        # g.set_titles(col_template="max_pts = {col_name}")

        # # Adjust the figure so the title doesn't get cut off and add some space
        # g.figure.subplots_adjust(top=1.0, wspace=0.3)  # Increased top and subplot spacing
        # g.figure.suptitle('Overlap Area by num_thresh and num_dirs for different max_pts values (close to 0 is best)', y=0.97)

        # # Save plot to the PDF
        # plt.tight_layout()
        # pdf.savefig(g.figure)
        # plt.close(g.figure)








        # # Create DataFrame for silhouette score plotting
        # df_silhouette = pd.DataFrame({
        #     'num_dirs': num_dirs_vals,
        #     'num_thresh': num_thresh_vals,
        #     'max_pts': max_pts_vals,
        #     'silhouette_score': s_scores  # Use the correct column name here
        # })

        # # Convert num_thresh and num_dirs to categorical for better plotting
        # df_silhouette['num_thresh'] = df_silhouette['num_thresh'].astype('category')
        # df_silhouette['num_dirs'] = df_silhouette['num_dirs'].astype('category')

        # # Get unique values for ordering
        # thresh_order = df_silhouette['num_thresh'].unique()
        # dirs_order = df_silhouette['num_dirs'].unique()

        # # Create a FacetGrid for each max_pts value with a larger figure size to prevent squishing
        # g = sns.FacetGrid(df_silhouette, col="max_pts", height=5, aspect=1.5)

        # # Use a barplot to show silhouette score as a function of num_thresh and num_dirs
        # g.map(sns.barplot, "num_thresh", "silhouette_score", "num_dirs", order=thresh_order, hue_order=dirs_order, palette="viridis")

        # # Adjust titles and labels
        # g.add_legend(title="num_dirs")
        # g.set_axis_labels("num_thresh", "Silhouette Score")
        # g.set_titles(col_template="max_pts = {col_name}")

        # # Adjust the figure so the title doesn't get cut off and add some space
        # g.figure.subplots_adjust(top=1.0, wspace=0.3)  # Increased top and subplot spacing
        # g.figure.suptitle('Silhouette Score by num_thresh and num_dirs for different max_pts values (close to 1.0 is best)', y=0.97)

        # # Save plot to the PDF
        # plt.tight_layout()
        # pdf.savefig(g.figure)
        # plt.close(g.figure)



        # Font size variables
        title_fontsize = 20
        label_fontsize = 16
        annot_fontsize = 14
        cbar_fontsize = 14
        suptitle_fontsize = 24
        
        max_pts_vals, num_thresh_vals, num_dirs_vals = zip(*params_data)

        # Create DataFrame for silhouette score plotting
        df_silhouette = pd.DataFrame({
            'num_dirs': num_dirs_vals,
            'num_thresh': num_thresh_vals,
            'max_pts': max_pts_vals,
            'silhouette_score': s_scores  # Use the correct column name here
        })

        # Convert num_thresh and num_dirs to categorical for better plotting
        df_silhouette['num_thresh'] = df_silhouette['num_thresh'].astype('category')
        df_silhouette['num_dirs'] = df_silhouette['num_dirs'].astype('category')

        # Create a list of unique max_pts values
        unique_max_pts = df_silhouette['max_pts'].unique()

        
        # Adjust width_ratios depending on the number of unique max_pts values
        if len(unique_max_pts) > 1:
            gridspec_kw = {'width_ratios': [1] * len(unique_max_pts) + [0.05]}  # Add extra space for color bar
        else:
            gridspec_kw = {'width_ratios': [1]}  # No color bar if there's only one plot

        # Create a figure with subplots (one for each max_pts)
        fig, axes = plt.subplots(1, len(unique_max_pts), figsize=(45, 15), sharey=True, gridspec_kw=gridspec_kw)

        # Ensure axes is always treated as an iterable, even if there's only one plot
        if len(unique_max_pts) == 1:
            axes = [axes]

        vlag_cmap_reversed = sns.color_palette("coolwarm", as_cmap=True).reversed()

        # Calculate the global min and max silhouette score for the entire dataset
        vmin = df_silhouette['silhouette_score'].min()
        vmax = df_silhouette['silhouette_score'].max()

        # Iterate over each unique max_pts value and create a heatmap
        for i, max_pts in enumerate(unique_max_pts):
            ax = axes[i]
    
            # Filter the data for the current max_pts value
            df_filtered = df_silhouette[df_silhouette['max_pts'] == max_pts]
            
            # Debugging print to see the filtered DataFrame
            print(f"Filtered DataFrame for max_pts = {max_pts}")
            print(df_filtered.head())  # Check if df_filtered has valid data

            # Pivot the data to create a matrix format for the heatmap
            df_pivot = df_filtered.pivot(index="num_thresh", columns="num_dirs", values="silhouette_score")

            # Debugging print to see the pivoted DataFrame
            print(f"Pivoted DataFrame for max_pts = {max_pts}")
            print(df_pivot)
            
            if df_pivot.empty:
                print(f"Warning: No data available for max_pts = {max_pts}. Skipping heatmap.")
                continue

            # Create the heatmap
            sns.heatmap(df_pivot, annot=True, cmap=vlag_cmap_reversed, ax=ax, vmin=vmin, vmax=vmax, cbar=False, annot_kws={"size": annot_fontsize})

            # Set titles and labels with the updated font sizes
            ax.set_title(f'max_pts = {max_pts}', fontsize=title_fontsize)
            ax.set_xlabel('num_dirs', fontsize=label_fontsize)
            if i == 0:  # Only show y-label on the first subplot
                ax.set_ylabel('num_thresh', fontsize=label_fontsize)

        # Create the color bar on the far right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for color bar (left, bottom, width, height)
        sm = plt.cm.ScalarMappable(cmap=vlag_cmap_reversed, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=cbar_fontsize)

        # Set a common title
        fig.suptitle('Silhouette Score Heatmap by num_thresh and num_dirs for different max_pts values (close to 1.0 is best)', y=0.97, fontsize=suptitle_fontsize)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.9, 0.95]) 

        # Save plot to the PDF
        pdf.savefig(fig)
        plt.close(fig)
    print(f"Experiment parameter sweep complete. Results saved in {pdf_filename}")
