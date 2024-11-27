import os
import pandas as pd
from urllib.parse import urlparse

def ingest_DWC_and_save_urls(DWC_csv_or_txt_file, dir_home, accumulated_data):
    """
    Extract URLs from the file in the directory, and append domain and URL pairs to the accumulated data.
    
    Args:
    - DWC_csv_or_txt_file: The name of the file containing the URLs.
    - dir_home: Directory where the file is located.
    - accumulated_data: List to store the domain and URL pairs.
    
    Returns:
    - None (the accumulated_data is updated in place).
    """
    file_path = os.path.join(dir_home, DWC_csv_or_txt_file)
    file_extension = DWC_csv_or_txt_file.split('.')[-1]

    try:
        # Read the file based on its extension
        if file_extension == 'txt':
            df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
        elif file_extension == 'csv':
            try:
                df = pd.read_csv(file_path, sep=",", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_path, sep="|", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
                    except pd.errors.ParserError:
                        df = pd.read_csv(file_path, sep=";", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
        else:
            print(f"DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened")
            return None
    except Exception as e:
        print(f"Error while reading file: {e}")
        return None

    # Check if 'identifier' column exists in the dataframe
    if 'identifier' not in df.columns:
        print(f"'identifier' column not found in the file {DWC_csv_or_txt_file}")
        return None

    # Extract URLs from 'identifier' column
    urls = df['identifier'].dropna().tolist()

    # Iterate through URLs and extract domains
    for url in urls:
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc  # Extract the domain (netloc)
            if domain and not any(domain == entry[0] for entry in accumulated_data):  # Check if domain is already in the list
                accumulated_data.append((domain, url))  # Store domain and URL as a tuple
        except Exception as e:
            print(f"Error parsing URL {url}: {e}")

def save_accumulated_data_to_csv(accumulated_data, path_to_csv):
    """
    Save the accumulated domain-URL pairs to a CSV file.
    
    Args:
    - accumulated_data: List of domain-URL pairs.
    - path_to_csv: The file path where the data should be saved.
    
    Returns:
    - None
    """
    # Convert the list to a DataFrame for saving
    filtered_urls_df = pd.DataFrame(accumulated_data, columns=['domain', 'url'])

    # Save the filtered domains and URLs to a CSV file
    filtered_urls_df.to_csv(path_to_csv, index=False)
    print(f"Filtered URLs saved to {path_to_csv}")


if __name__ == "__main__":
    # List of directories to process
    dir_homes = [
        "/media/nas/GBIF_Downloads/Magnoliales/Annonaceae",
        "/media/nas/GBIF_Downloads/Magnoliales/Degeneriaceae",
        "/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae",
        "/media/nas/GBIF_Downloads/Magnoliales/Himantandraceae",
        "/media/nas/GBIF_Downloads/Magnoliales/Magnoliaceae",
        "/media/nas/GBIF_Downloads/Magnoliales/Myristicaceae",
        "/media/nas/GBIF_Downloads/Cornales/Cornaceae",
        "/media/nas/GBIF_Downloads/Cornales/Hydrangeaceae",
        "/media/nas/GBIF_Downloads/Cornales/Loasaceae",
        "/media/nas/GBIF_Downloads/Cornales/Nyssaceae",
        "/media/nas/GBIF_Downloads/Dipsacales/Caprifoliaceae",
        "/media/nas/GBIF_Downloads/Dipsacales/Viburnaceae",
        "/media/nas/GBIF_Downloads/Ericales/Actinidiaceae",
        "/media/nas/GBIF_Downloads/Ericales/Balsaminaceae",
        "/media/nas/GBIF_Downloads/Ericales/Clethraceae",
        "/media/nas/GBIF_Downloads/Ericales/Cyrillaceae",
        "/media/nas/GBIF_Downloads/Ericales/Diapensiaceae",
        "/media/nas/GBIF_Downloads/Ericales/Ebenaceae",
        "/media/nas/GBIF_Downloads/Ericales/Ericaceae",
        "/media/nas/GBIF_Downloads/Ericales/Fouquieriaceae",
        "/media/nas/GBIF_Downloads/Ericales/Lecythidaceae",
        "/media/nas/GBIF_Downloads/Ericales/Marcgraviaceae",
        "/media/nas/GBIF_Downloads/Ericales/Mitrastemonaceae",
        "/media/nas/GBIF_Downloads/Ericales/Pentaphylacaceae",
        "/media/nas/GBIF_Downloads/Ericales/Polemoniaceae",
        "/media/nas/GBIF_Downloads/Ericales/Primulaceae",
        "/media/nas/GBIF_Downloads/Ericales/Roridulaceae",
        "/media/nas/GBIF_Downloads/Ericales/Sapotaceae",
        "/media/nas/GBIF_Downloads/Ericales/Sarraceniaceae",
        "/media/nas/GBIF_Downloads/Ericales/Sladeniaceae",
        "/media/nas/GBIF_Downloads/Ericales/Styracaceae",
        "/media/nas/GBIF_Downloads/Ericales/Symplocaceae",
        "/media/nas/GBIF_Downloads/Ericales/Tetrameristaceae",
        "/media/nas/GBIF_Downloads/Ericales/Theaceae",
        "/media/nas/GBIF_Downloads/Fagales/Betulaceae",
        "/media/nas/GBIF_Downloads/Fagales/Casuarinaceae",
        "/media/nas/GBIF_Downloads/Fagales/Fagaceae",
        "/media/nas/GBIF_Downloads/Fagales/Juglandaceae",
        "/media/nas/GBIF_Downloads/Fagales/Myricaceae",
        "/media/nas/GBIF_Downloads/Fagales/Nothofagaceae",
        "/media/nas/GBIF_Downloads/Fagales/Ticodendraceae",
        "/media/nas/GBIF_Downloads/Moraceae",
    ]
    
    domain_list_path = "multimedia.txt"  # The name of the file containing the URLs
    path_to_csv = "/media/nas/GBIF_Downloads/one_url_per_domain.csv"  # Output file

    # List to accumulate the domain-URL pairs across all directories
    accumulated_data = []

    # Loop through each directory and process the file
    for dir_home in dir_homes:
        print(f"Processing directory: {dir_home}")
        ingest_DWC_and_save_urls(domain_list_path, dir_home, accumulated_data)
    
    # Save the accumulated data to a single CSV
    save_accumulated_data_to_csv(accumulated_data, path_to_csv)