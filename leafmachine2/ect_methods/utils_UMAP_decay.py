import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
from scipy.spatial import procrustes
from sklearn.metrics import silhouette_score
import warnings
from tqdm import tqdm

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from leafmachine2.ect_methods.utils_metrics import compute_metrics
from leafmachine2.ect_methods.utils_UMAP import compute_umap, flatten_core_metrics

def compute_statistical_significance(baseline_metrics, comparison_metrics):
    """
    Compute statistical significance between baseline and comparison metrics.
    
    Parameters:
    -----------
    baseline_metrics : array-like
        Metrics from the full dataset (fraction = 1.0)
    comparison_metrics : array-like
        Metrics from the subsampled dataset
        
    Returns:
    --------
    dict
        Dictionary containing p-value and effect size
    """
    _, p_value = stats.mannwhitneyu(baseline_metrics, comparison_metrics, alternative='two-sided')
    effect_size = abs(np.mean(baseline_metrics) - np.mean(comparison_metrics)) / np.std(baseline_metrics)
    return {'p_value': p_value, 'effect_size': effect_size}

def parallel_umap_computation(ect_data, indices, output_path2D, output_path3D):
    """
    Wrapper function for parallel UMAP computation
    """
    subset_ect_data = [ect_data[i] for i in indices]
    return compute_umap(output_path2D, output_path3D, subset_ect_data, indices)

def test_umap_decay_with_metrics(ect_data, group_labels, shapes, component_names, save_dir, 
                                labels_fullname, labels_genus, labels_family, overall_family,
                                n_iterations=5, n_jobs=-1):
    """
    Enhanced UMAP decay analysis with statistical testing and parallel processing
    
    Additional Parameters:
    ---------------------
    n_iterations : int
        Number of iterations for each fraction to assess stability
    n_jobs : int
        Number of parallel jobs to run (-1 for all available cores)
    """
    output_npz_path2D = os.path.join(save_dir, "UMAP_2D.npz")
    output_npz_path3D = os.path.join(save_dir, "UMAP_3D.npz")
    fractions = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05]
    
    all_core_metrics = []
    all_comparison_metrics = []
    all_stability_metrics = []
    
    # Compute baseline (full dataset) metrics first
    print("Computing baseline metrics with full dataset...")
    baseline_scores2D, baseline_scores3D = compute_umap(output_npz_path2D, output_npz_path3D, 
                                                      ect_data, np.arange(len(ect_data)))
    baseline_core, baseline_comparison = compute_metrics(
        save_dir, baseline_scores2D, baseline_scores3D, 
        labels_fullname, labels_genus, overall_family, "UMAP_baseline")
    
    for frac in tqdm(fractions, desc="Processing fractions"):
        subset_size = int(len(ect_data) * frac)
        iteration_results = []
        
        # Parallel processing for multiple iterations
        def process_iteration(iteration):
            np.random.seed(2024 + iteration)  # Ensure reproducibility with different seeds
            subset_indices = np.random.choice(len(ect_data), subset_size, replace=False).astype(int)
            subset_indices.sort()
            
            subset_output_npz_path2D = output_npz_path2D.replace('.npz', f'__{int(frac * 100):03}_{iteration}.npz')
            subset_output_npz_path3D = output_npz_path3D.replace('.npz', f'__{int(frac * 100):03}_{iteration}.npz')
            
            # Compute UMAP and metrics
            umap_scores2D, umap_scores3D = parallel_umap_computation(
                ect_data, subset_indices, subset_output_npz_path2D, subset_output_npz_path3D)
            
            subset_labels_fullname = [labels_fullname[i] for i in subset_indices]
            subset_labels_genus = [labels_genus[i] for i in subset_indices]
            
            core_metrics, comparison_metrics = compute_metrics(
                save_dir, umap_scores2D, umap_scores3D, 
                subset_labels_fullname, subset_labels_genus, overall_family, 
                f"UMAP__{int(frac * 100):03}_{iteration}")
            
            # Compute stability metrics
            stability_metrics = {
                'procrustes_2D': procrustes(baseline_scores2D[subset_indices], umap_scores2D)[2],
                'procrustes_3D': procrustes(baseline_scores3D[subset_indices], umap_scores3D)[2],
                'silhouette_2D': silhouette_score(umap_scores2D, subset_labels_genus),
                'silhouette_3D': silhouette_score(umap_scores3D, subset_labels_genus)
            }
            
            return core_metrics, comparison_metrics, stability_metrics
        
        # Run parallel iterations
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_iteration)(i) for i in range(n_iterations))
        
        # Process results
        for core_metrics, comparison_metrics, stability_metrics in results:
            flat_core_metrics = flatten_core_metrics(core_metrics, int(frac * 100))
            all_core_metrics.append(pd.DataFrame([flat_core_metrics]))
            
            comparison_metrics_df = pd.DataFrame(comparison_metrics)
            comparison_metrics_df['fraction'] = int(frac * 100)
            all_comparison_metrics.append(comparison_metrics_df)
            
            stability_metrics['fraction'] = int(frac * 100)
            all_stability_metrics.append(pd.DataFrame([stability_metrics]))
    
    # Combine all metrics
    combined_core_metrics_df = pd.concat(all_core_metrics, ignore_index=True)
    combined_comparison_metrics_df = pd.concat(all_comparison_metrics, ignore_index=True)
    combined_stability_metrics_df = pd.concat(all_stability_metrics, ignore_index=True)
    
    # Save results
    save_results(save_dir, combined_core_metrics_df, combined_comparison_metrics_df, 
                combined_stability_metrics_df)
    
    # Create enhanced visualizations
    create_enhanced_visualizations(save_dir, combined_core_metrics_df, 
                                 combined_comparison_metrics_df, 
                                 combined_stability_metrics_df)
    
    return combined_core_metrics_df, combined_comparison_metrics_df, combined_stability_metrics_df

def save_results(save_dir, core_metrics_df, comparison_metrics_df, stability_metrics_df):
    """Save all results to CSV files"""
    core_metrics_df.to_csv(os.path.join(save_dir, "UMAP_core_metrics_comparison.csv"), index=False)
    comparison_metrics_df.to_csv(os.path.join(save_dir, "UMAP_comparison_metrics_comparison.csv"), index=False)
    stability_metrics_df.to_csv(os.path.join(save_dir, "UMAP_stability_metrics.csv"), index=False)

def create_enhanced_visualizations(save_dir, core_metrics_df, comparison_metrics_df, stability_metrics_df):
    """Create enhanced visualizations with confidence intervals and statistical significance"""
    # Set style
    plt.style.use('seaborn')
    
    # 1. Core metrics visualization
    metrics = core_metrics_df.columns.drop("fraction")
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=core_metrics_df, x='fraction', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} vs Data Fraction')
        axes[i].set_xlabel('Fraction of Data (%)')
        axes[i].set_ylabel('Metric Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "UMAP_core_metrics_boxplot.png"))
    plt.close()
    
    # 2. Stability metrics visualization
    stability_metrics = stability_metrics_df.columns.drop("fraction")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(stability_metrics):
        sns.boxplot(data=stability_metrics_df, x='fraction', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} vs Data Fraction')
        axes[i].set_xlabel('Fraction of Data (%)')
        axes[i].set_ylabel('Metric Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "UMAP_stability_metrics_boxplot.png"))
    plt.close()
    
    # 3. Create summary heatmap
    plt.figure(figsize=(12, 8))
    summary_data = stability_metrics_df.groupby('fraction').mean()
    sns.heatmap(summary_data, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Stability Metrics Summary Heatmap')
    plt.savefig(os.path.join(save_dir, "UMAP_stability_heatmap.png"))
    plt.close()










import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
from leafmachine2.ect_methods.utils_UMAP import compute_umap, compute_umap_direct
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines

def sample_umap_sensitivity(ect_data, save_path_UMAP_Decay, fractions=[1.0, 0.75, 0.5, 0.25, 0.15, 0.1, 0.05], n_replicates=10, n_jobs=4):
    os.makedirs(save_path_UMAP_Decay, exist_ok=True)

    # Compute baseline UMAP on full dataset for reference
    baseline_2D, baseline_3D = compute_umap_direct(ect_data)
    
    results = []
    
    for frac in tqdm(fractions, desc="Processing fractions"):
        subset_size = int(len(ect_data) * frac)
        
        def process_replicate(iteration):
            print(f"{int(frac * 100):03}__{iteration}/{n_replicates}")
            np.random.seed(2024 + iteration)
            subset_indices = np.random.choice(len(ect_data), subset_size, replace=False)
            umap_scores2D, umap_scores3D = compute_umap_direct(ect_data, subset_indices=subset_indices, seed=2024 + iteration)

            # Calculate Euclidean distance between subset and baseline UMAP embeddings
            euclidean_dist_2D = np.linalg.norm(baseline_2D[subset_indices] - umap_scores2D, axis=1)
            euclidean_dist_3D = np.linalg.norm(baseline_3D[subset_indices] - umap_scores3D, axis=1)

            # Calculate RMSE, MAE, and MAD for both 2D and 3D
            rmse_2D = np.sqrt(mean_squared_error(baseline_2D[subset_indices], umap_scores2D))
            mae_2D = mean_absolute_error(baseline_2D[subset_indices], umap_scores2D)
            mad_2D = np.mean(np.abs(euclidean_dist_2D - np.mean(euclidean_dist_2D)))

            rmse_3D = np.sqrt(mean_squared_error(baseline_3D[subset_indices], umap_scores3D))
            mae_3D = mean_absolute_error(baseline_3D[subset_indices], umap_scores3D)
            mad_3D = np.mean(np.abs(euclidean_dist_3D - np.mean(euclidean_dist_3D)))
            
            return {
                "fraction": frac, "iteration": iteration,
                "rmse_2D": rmse_2D, "mae_2D": mae_2D, "mad_2D": mad_2D,
                "rmse_3D": rmse_3D, "mae_3D": mae_3D, "mad_3D": mad_3D
            }
        
        # Run the replicate function in parallel
        results += Parallel(n_jobs=n_jobs)(delayed(process_replicate)(i) for i in range(n_replicates))

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_path_UMAP_Decay, "UMAP_sensitivity_metrics.csv"), index=False)
    print(results_df)

    # Plot sensitivity analysis to determine MSDE
    plot_significance_distributions(results_df, save_path_UMAP_Decay, fractions)

def plot_significance_distributions(results_df, save_path, fractions):
    """
    Plots the minimal statistically detectable effect (MSDE) based on RMSE, MAE, and MAD for 2D and 3D embeddings.
    """

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ["rmse_2D", "mae_2D", "mad_2D", "rmse_3D", "mae_3D", "mad_3D"]
    titles = ["RMSE 2D", "MAE 2D", "MAD 2D", "RMSE 3D", "MAE 3D", "MAD 3D"]

    # Obtain baseline values from the 100% fraction in results_df
    baseline_means = results_df[results_df["fraction"] == 1.0].groupby("fraction").mean().iloc[0]

    for ax, metric, title in zip(axs.flatten(), metrics, titles):
        # Compute mean metric values for each fraction
        metric_means = results_df.groupby("fraction")[metric].mean()
        
        # Plot the metric values across fractions
        ax.plot(fractions, metric_means, marker="o", label=f"Mean {metric}")
        
        # Plot the baseline value as a horizontal line (100% fraction)
        ax.axhline(y=baseline_means[metric], color="r", linestyle="--", label=f"Baseline {metric}")

        ax.set_title(f"{title} vs. Fraction of Data")
        ax.set_xlabel("Fraction of Data")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.suptitle("Minimal Statistically Detectable Effect (MSDE) Across Fractions")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "MSDE_sensitivity_analysis.png"))
    plt.show()


# def sample_umap_sensitivity(ect_data, save_path_UMAP_Decay, fractions=[1.0, 0.75, 0.5, 0.25, 0.15, 0.1, 0.05], n_replicates=100, n_jobs=4):
#     os.makedirs(save_path_UMAP_Decay, exist_ok=True)

#     baseline_2D, baseline_3D = compute_umap_direct(ect_data)
    
#     results = []
    
#     for frac in tqdm(fractions, desc="Processing fractions"):
#         subset_size = int(len(ect_data) * frac)
        
#         def process_replicate(iteration):
#             print(f"{int(frac * 100):03}__{iteration}/{n_replicates}")
#             np.random.seed(2024 + iteration)
#             subset_indices = np.random.choice(len(ect_data), subset_size, replace=False)
#             umap_scores2D, umap_scores3D = compute_umap_direct(ect_data, subset_indices=subset_indices, seed=2024 + iteration)

#             euclidean_dist_2D = np.linalg.norm(baseline_2D[subset_indices] - umap_scores2D, axis=1)
#             euclidean_dist_3D = np.linalg.norm(baseline_3D[subset_indices] - umap_scores3D, axis=1)

#             rmse_2D = np.sqrt(mean_squared_error(baseline_2D[subset_indices], umap_scores2D))
#             mae_2D = mean_absolute_error(baseline_2D[subset_indices], umap_scores2D)
#             mad_2D = np.mean(np.abs(euclidean_dist_2D - np.mean(euclidean_dist_2D)))

#             rmse_3D = np.sqrt(mean_squared_error(baseline_3D[subset_indices], umap_scores3D))
#             mae_3D = mean_absolute_error(baseline_3D[subset_indices], umap_scores3D)
#             mad_3D = np.mean(np.abs(euclidean_dist_3D - np.mean(euclidean_dist_3D)))
            
#             return {
#                 "fraction": frac, "iteration": iteration,
#                 "rmse_2D": rmse_2D, "mae_2D": mae_2D, "mad_2D": mad_2D,
#                 "rmse_3D": rmse_3D, "mae_3D": mae_3D, "mad_3D": mad_3D
#             }
        
#         results += Parallel(n_jobs=n_jobs)(delayed(process_replicate)(i) for i in range(n_replicates))

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(os.path.join(save_path_UMAP_Decay, "UMAP_sensitivity_metrics.csv"), index=False)

#     baseline_values = {
#         "rmse_2D": results_df["rmse_2D"].mean(),
#         "mae_2D": results_df["mae_2D"].mean(),
#         "mad_2D": results_df["mad_2D"].mean(),
#         "rmse_3D": results_df["rmse_3D"].mean(),
#         "mae_3D": results_df["mae_3D"].mean(),
#         "mad_3D": results_df["mad_3D"].mean()
#     }

#     # plot_sensitivity_analysis(results_df, save_path_UMAP_Decay)
#     plot_significance_distributions(results_df, save_path_UMAP_Decay, baseline_values, fractions)

# def plot_sensitivity_analysis(results_df, save_dir):
#     metrics = ["rmse", "mae", "mad"]
#     dimensions = ["2D", "3D"]
    
#     for metric in metrics:
#         for dim in dimensions:
#             plt.figure()
#             for frac, group in results_df.groupby("fraction"):
#                 plt.plot(group["iteration"], group[f"{metric}_{dim}"], label=f"{int(frac * 100)}%")
#             plt.title(f"{metric.upper()} {dim} vs Iteration")
#             plt.xlabel("Iteration")
#             plt.ylabel(f"{metric.upper()} ({dim})")
#             plt.legend(title="Fraction of Data")
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f"{metric}_{dim}_sensitivity.png"))
#             plt.close()
    

# def plot_significance_distributions(results_df, save_dir, baseline_values, fractions):
#     metrics = ["rmse", "mae", "mad"]
#     dimensions = ["2D", "3D"]
    
#     for metric in metrics:
#         for dim in dimensions:
#             plt.figure(figsize=(10, 6))
            
#             palette = sns.color_palette("tab20", n_colors=len(fractions))
#             color_map = dict(zip(fractions, palette))
            
#             for frac in fractions:
#                 subset = results_df[results_df['fraction'] == frac]
#                 sns.kdeplot(
#                     data=subset, 
#                     x=f"{metric}_{dim}", 
#                     color=color_map[frac], 
#                     fill=True, 
#                     alpha=0.5, 
#                     label=f"{int(frac * 100)}%"
#                 )
            
#             baseline = baseline_values.get(f"{metric}_{dim}", None)
#             if baseline is not None:
#                 plt.axvline(baseline, color="black", linestyle="--", label="Baseline")
            
#             plt.title(f"Distribution of {metric.upper()} ({dim}) across Fractions")
#             plt.xlabel(f"{metric.upper()} ({dim})")
#             plt.ylabel("Density")
#             plt.grid(True, alpha=0.3)
#             plt.legend(title="Fraction of Data", loc="upper right", frameon=True)
#             plt.savefig(os.path.join(save_dir, f"{metric}_{dim}_distribution.png"))
#             plt.close()