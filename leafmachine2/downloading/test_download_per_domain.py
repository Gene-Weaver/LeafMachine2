import os
import pandas as pd
from test_proxy import download_image_directly  # Assuming the function is in test_proxy.py

def test_urls_in_file(file_path, target_dir, output_csv=None):
    """
    This function extracts domains and URLs from a file and tests downloading each of them using `download_image_directly`.
    Logs success/failure, response status, and errors in a CSV file.
    
    Args:
    - file_path: The path to the input txt/csv file containing the domains and URLs.
    - target_dir: Directory where the images will be saved.
    - output_csv: Optional, path to save the test report CSV file. If None, it saves it in the same location as the input file with '_tested.csv' appended.
    
    Returns:
    - None
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Determine output CSV name if not provided
    if output_csv is None:
        base_name, ext = os.path.splitext(file_path)
        output_csv = f"{base_name}_tested.csv"

    # Load the file (txt or csv)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Ensure the file contains 'domain' and 'url' columns
    if 'url' not in df.columns or 'domain' not in df.columns:
        print(f"File does not contain 'domain' and 'url' columns.")
        return
    
    # Extract the domains and URLs to test
    domains = df['domain'].dropna().tolist()
    urls = df['url'].dropna().tolist()

    # Initialize a dictionary to store the test results
    test_results = {
        'domain': [],  # Domain from input CSV
        'url': [],     # Original URL
        'status': [],  # Log success or failure
        'error': [],   # Log any errors
        'ind': [],     # Index of the URL in the list
    }

    # Test each URL
    for ind, (domain, url) in enumerate(zip(domains, urls)):
        print(f"{ind+1} // {len(urls)}")
        try:
            print(f"Testing URL: {url}")
            fname = f"image_{ind}"  # Use the index to dynamically generate the filename
            
            # Try to download the image using the provided function and get the status
            status, message = download_image_directly(url, target_dir, fname)
            
            # Log the domain and URL
            test_results['domain'].append(domain)
            test_results['url'].append(url)
            
            # Log the status and any message
            test_results['status'].append(status)
            test_results['error'].append(message if message else "None")
            test_results['ind'].append(ind)

        except Exception as e:
            # Log unexpected errors during the process
            test_results['domain'].append(domain)
            test_results['url'].append(url)
            test_results['status'].append("Failure")
            test_results['error'].append(str(e))
            test_results['ind'].append(ind)

    # Save the results to a new CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Test results saved to {output_csv}")


if __name__ == "__main__":
    # Example usage:
    input_file = 'D:/Dropbox/LeafMachine2/leafmachine2/downloading/one_url_per_domain.csv' 
    target_dir = 'D:/Dropbox/LeafMachine2/leafmachine2/downloading/one_url_per_domain_test'
    test_urls_in_file(input_file, target_dir)
