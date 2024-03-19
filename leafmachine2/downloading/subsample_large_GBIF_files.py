import os
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

"""
This will subsample GBIF files that have millions of rows.

Go to GBIF, make your query, download the DWC Archive version.

We will use the occurence and image files.

* This will allow you to subsample the occ file by unique entries in a column
* It will let yo udetermine how many rows per instance to keep
* Then it will match the rows to the images file to create the combined file that is required to download the images

"""

def sample_rows_from_csv(filename, column_name, n_rows, output_file='output.csv'):
    with Client() as client: 
        # Setting dtype for the entire dataframe as 'object'
        df = dd.read_table(filename, delimiter='\t', assume_missing=True, on_bad_lines='skip', dtype='object')
        headers = df.head(1).columns.tolist()
        print(headers)

        # Sample the dataframe by grouping and then sampling
        sampled_df = df.groupby(column_name).apply(lambda x: x.sample(min(len(x), n_rows)), meta=df)

        with ProgressBar():
            sampled_df.compute().to_csv(output_file, index=False)

def sample_rows_from_combined_columns(filename, columns_to_groupby, new_column_name, separator="_", n_rows=10, output_file='output.csv'):
    
    with Client() as client:
        # Setting dtype for the entire dataframe as 'object'
        df = dd.read_table(filename, delimiter='\t', assume_missing=True, on_bad_lines='skip', dtype='object')
        headers = df.head(1).columns.tolist()
        print(headers)
        
        # Convert columns_to_groupby to string in a more optimized manner
        df[columns_to_groupby] = df[columns_to_groupby].astype(str)
        
        # Create a new column by combining the desired columns
        df[new_column_name] = df[columns_to_groupby].apply(lambda row: separator.join(row.dropna()), axis=1, meta=('string'))

        # Group by the new column and sample
        sampled_df = df.groupby(new_column_name).apply(lambda x: x.sample(min(len(x), n_rows)), meta=df)
        
        with ProgressBar():
            sampled_df.compute().to_csv(output_file, index=False)
        

def match_rows_by_id(subset_file, images_file, output_file='output_images.csv'):
    client = Client()
    subset_df = dd.read_csv(subset_file, assume_missing=True)
    images_df = dd.read_csv(images_file, assume_missing=True)
    merged_df = subset_df.merge(images_df, on="id", how="inner")
    with ProgressBar():
        merged_df.compute().to_csv(output_file, index=False)
    client.close()

if __name__ == "__main__":
    dir_GBIF = 'D:\Dropbox\LM2_Env\Image_Datasets\GBIF_Ingest_2023\DWC_wLoc'  # Update this with base directory
    dir_OUT = 'D:\Dropbox\LM2_Env\Image_Datasets\GBIF_Ingest_2023\DWC_wLoc_subset_herb'  # Update this with output directory

    if not os.path.exists(dir_OUT):
        os.makedirs(dir_OUT)

    version = "sample_single" # FROM "sample_single" OR "sample_mult"
    new_column_name = 'fullname'

    # For 'sample' operation
    filename = os.path.join(dir_GBIF, 'occurrence.txt')
    column_name = 'institutionCode'
    n_rows = 10
    output_sample_file = os.path.join(dir_OUT, 'occ.csv')

    # For 'match' operation
    images_file = os.path.join(dir_GBIF, 'multimedia.txt')
    output_match_file = os.path.join(dir_OUT, 'img.csv')

    # File to save the combined data
    filename_combined = os.path.join(dir_OUT, 'combined.csv')

    if version == "sample_single":
        sample_rows_from_csv(filename, column_name, n_rows, output_sample_file)
        match_rows_by_id(output_sample_file, images_file, output_match_file)
    else:
        #OR is trying to subset by taxa
        columns_to_groupby = ['collectionID','datasetID', 'institutionCode', 'collectionCode',] # ["family", "genus", "specific_epiphet"]
        sample_rows_from_combined_columns(filename, columns_to_groupby, new_column_name, n_rows=n_rows, output_file=output_sample_file)
        match_rows_by_id(output_sample_file, images_file, output_match_file)

    # Combine both dataframes using pandas
    occ_new = pd.read_csv(output_sample_file)
    img_new = pd.read_csv(output_match_file)
    img_new = img_new.rename(columns={"identifier": "url"})
    combined = pd.concat([occ_new, img_new], axis=1, sort=False)

    combined.to_csv(filename_combined, index=False)
