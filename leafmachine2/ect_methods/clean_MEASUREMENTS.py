import os, sys

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
try:
    from leafmachine2.ect_methods.utils_ect import preprocessing
except:
    from utils_ect import preprocessing


if __name__ == '__main__':
    # file_path = "G:/Thais/LM2/Data/Measurements/LM2_MEASUREMENTS.csv"
    # outline_path = "G:/Thais/LM2/Keypoints/Simple_Labels"
    # path_figure = "G:/Thais/LM2/Data/Measurements/CF_Plot_Disagreement.png"

    file_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_10_30__19-14-22/Data/Measurements/LM2_MEASUREMENTS.csv"
    outline_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_10_30__19-14-22/Keypoints/Simple_Labels"
    path_figure = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_10_30__19-14-22/Keypoints/CF_Plot_Disagreement.png"
    
    clean_file_path = file_path.replace("LM2_MEASUREMENTS.csv", "LM2_MEASUREMENTS_CLEAN.csv")


    cleaned_df = preprocessing(file_path, outline_path, 
                               show_CF_plot=False, show_first_raw_contour=False, show_df_head=False, is_Thais=False,
                               path_figure=path_figure)
    cleaned_df.to_csv(clean_file_path, index=False)
