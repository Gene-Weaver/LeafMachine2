import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
import pandas as pd

def visualize_EFDs(filename):
    # read the EFD coefficients and parameters from the CSV file
    shapes_coeffs = []
    shapes_params = []
    filenames = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # if i >= 10:
            #     break
            for j in range(1, 40):
                if row['coeffs_0'] != 'NA':
                    if 'leaf' in row['annotation_name'].split('_'):
                        coeffs = row[f'coeffs_{j}'].replace('[', '').replace(']', '').split()
                        for k in range(0,4):
                            coeffs[k] = float(coeffs[k])
                        params = {'a0': float(row['efd_a0']), 'c0': float(row['efd_c0']),
                                'scale': float(row['efd_scale']), 'angle': float(row['efd_angle']),
                                'phase': float(row['efd_phase'])}
                        shapes_coeffs.append(coeffs)
                        shapes_params.append(params)
                        filename = ' '.join(row['filename'].split('_')[2:5])
                        filenames.append(filename)

    # assign colors based on unique filenames
    unique_filenames = set(filenames)
    cmap = plt.get_cmap('tab20')
    colors = {filename: cmap(i/len(unique_filenames)) for i, filename in enumerate(unique_filenames)}

    # plot the EFD coefficients for each shape
    for i, (coeffs, params) in enumerate(zip(shapes_coeffs, shapes_params)):
        harmonic_order = np.arange(len(coeffs))
        filename = filenames[i]
        color = colors[filename]
        plt.plot(harmonic_order, coeffs, color=color, linewidth=1, label=f'{filename} - Shape {i}')


    # create legend on right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Harmonic order')
    plt.ylabel('EFD coefficients')
    plt.gcf().set_size_inches(20, 20)
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\EFD_plot.png', dpi=300)




def visualize_EFDs_PCA(filename):
    # read the EFD coefficients and parameters from the CSV file
    shapes_coeffs = []
    shapes_params = []
    filenames = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # if i >= 10:
            #     break
            for j in range(1, 40):
                if row['coeffs_0'] != 'NA':
                    if 'leaf' in row['annotation_name'].split('_'):
                        coeffs = row[f'coeffs_{j}'].replace('[', '').replace(']', '').split()
                        for k in range(0,4):
                            coeffs[k] = float(coeffs[k])
                        params = {'a0': float(row['efd_a0']), 'c0': float(row['efd_c0']),
                                'scale': float(row['efd_scale']), 'angle': float(row['efd_angle']),
                                'phase': float(row['efd_phase'])}
                        shapes_coeffs.append(coeffs)
                        shapes_params.append(params)
                        filename = ' '.join(row['filename'].split('_')[2:5])
                        filenames.append(filename)

    # perform PCA on the EFD coefficients
    pca = PCA(n_components=4)
    pca.fit(shapes_coeffs)
    transformed = pca.transform(shapes_coeffs)

    # create DataFrame for parallel coordinates plot
    df = pd.DataFrame(transformed, columns=['PCA1', 'PCA2'])
    df['filename'] = filenames

    # plot parallel coordinates
    plt.figure(figsize=(20, 10))
    pd.plotting.parallel_coordinates(df, 'filename', color=plt.get_cmap('tab20')(np.linspace(0, 1, len(set(filenames)))))
    plt.xlabel('PCA dimension')
    plt.ylabel('PCA value')
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\EFD_pca_plot.png', dpi=300)


def visualize_EFDs_PCA_group(filename):
    # read the EFD coefficients and parameters from the CSV file
    shapes_coeffs = []
    shapes_params = []
    filenames = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            for j in range(1, 40):
                if row['coeffs_0'] != 'NA':
                    if 'leaf' in row['annotation_name'].split('_'):
                        coeffs = row[f'coeffs_{j}'].replace('[', '').replace(']', '').split()
                        for k in range(0,4):
                            coeffs[k] = float(coeffs[k])
                        params = {'a0': float(row['efd_a0']), 'c0': float(row['efd_c0']),
                                'scale': float(row['efd_scale']), 'angle': float(row['efd_angle']),
                                'phase': float(row['efd_phase'])}
                        shapes_coeffs.append(coeffs)
                        shapes_params.append(params)
                        filename = ' '.join(row['filename'].split('_')[2:5])
                        filenames.append(filename)

    # perform PCA on the EFD coefficients, using the filenames as the grouping variable
    pca = PCA(n_components=4)
    pca.fit(shapes_coeffs)
    transformed = pca.transform(shapes_coeffs)

    # create DataFrame for parallel coordinates plot
    df = pd.DataFrame(transformed, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    df['filename'] = filenames

    # plot parallel coordinates
    plt.figure(figsize=(40, 20))
    pd.plotting.parallel_coordinates(df, 'filename', color=plt.get_cmap('tab20')(np.linspace(0, 1, len(set(filenames)))))
    plt.xlabel('PCA dimension')
    plt.ylabel('PCA value')
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\EFD_pca_plot.png', dpi=300)

# Define a function to reconstruct shape from EFD coefficients and parameters
from scipy.interpolate import interp1d

def reconstruct_shape(coeffs, params, num_samples=300):
    a0 = params['a0']
    c0 = params['c0']
    scale = params['scale']
    angle = params['angle']
    phase = params['phase']
    N = len(coeffs)
    
    # Interpolate the coeffs array to a higher resolution
    interp_func = interp1d(np.arange(N), coeffs, kind='cubic', bounds_error=False)
    t = np.linspace(0, N-1, num_samples)
    
    # Generate the complex-valued signal
    Z = np.zeros(num_samples, dtype=np.complex)
    for n in range(num_samples):
        Z[n] = interp_func(n)
        for k in range(1, N//2):
            Z[n] += interp_func(k) * np.exp(1j * 2 * np.pi * k * n / N) + \
                    interp_func(-k) * np.exp(-1j * 2 * np.pi * k * n / N)
        Z[n] *= scale * np.exp(1j * (angle * n / N + phase))
        Z[n] += a0 + c0 * 1j * (n - N/2)
    return Z


def visualize_EFD_reconstruct(filename):
    shapes_coeffs = []
    shapes_params = []

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['coeffs_0'] != 'NA' and 'leaf' in row['annotation_name'].split('_'):
                coeffs = np.array([float(x) for x in row['coeffs_1'][1:-1].split()])
                params = {'a0': float(row['efd_a0']), 'c0': float(row['efd_c0']),
                          'scale': float(row['efd_scale']), 'angle': float(row['efd_angle']),
                          'phase': float(row['efd_phase'])}
                shapes_coeffs.append(coeffs)
                shapes_params.append(params)

    # Reconstruct all shapes
    reconstructed = []
    for coeffs, params in zip(shapes_coeffs, shapes_params):
        Z = reconstruct_shape(coeffs, params)
        reconstructed.append(np.array([(z.real, z.imag) for z in Z]))

    # Plot the reconstructed shapes
    fig, ax = plt.subplots(figsize=(10,10))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
    for i, shape in enumerate(reconstructed):
        color = colors[i % len(colors)]
        ax.plot(shape[:,0], shape[:,1], color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reconstructed Shapes')
    plt.savefig('D:\Dropbox\LM2_Env\Image_Datasets\Explore\Plots\EFD_pca_reconstruct_plot.png', dpi=300)



# example usage
visualize_EFD_reconstruct('D:\Dropbox\LM2_Env\Image_Datasets\TEST_LM2\Explore_single_whole_50p\Data\EFD/Explore_single_whole_50p_EFD.csv')
# example usage
visualize_EFDs('D:\Dropbox\LM2_Env\Image_Datasets\TEST_LM2\Explore_single_whole_50p\Data\EFD/Explore_single_whole_50p_EFD.csv')
