'''
This is similar to test_download_all_images_in_images_csv.py

This is used to parse the huge Magnoliopsida txt file into family batches saved in 
    /media/nas/GBIF_Downloads/Magnoliopsida_By_Family

Still uses the /media/nas/GBIF_Downloads/Magnoliopsida/multimedia.txt to house the image urls
'''


import os, math
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd

dask.config.set({'temporary_directory': '/data/tmp'})

def save_csv_by_family(file_path, output_dir, wishlist_csv_path, block_size=3000):
    """
    Processes a large Darwin Core (DWC) TXT or CSV file and saves subsets based on the 'family' column.

    Args:
        file_path (str): Name of the input large TXT or CSV file.
        output_dir (str): Directory where family-specific folders will be created and CSVs saved.
        block_size (int): Block size in MB for Dask to process the file in chunks.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    family_wishlist = pd.read_csv(wishlist_csv_path, sep=",", header=0, low_memory=False, dtype=str, on_bad_lines='skip')
    unique_families = family_wishlist['family'].dropna().unique().tolist()

    print("Number of unique families:", len(unique_families))

    ## Define the dtype for each column
    column_dtypes = {
        'filename': 'object',  # TEXT
        'image_height': 'float64',  # INTEGER
        'image_width': 'float64',  # INTEGER
        'component_name': 'object',  # TEXT
        'n_pts_in_polygon': 'float64',  # INTEGER
        'conversion_mean': 'float64',  # FLOAT
        'predicted_conversion_factor_cm': 'float64',  # FLOAT
        'area': 'float64',  # FLOAT
        'perimeter': 'float64',  # FLOAT
        'convex_hull': 'float64',  # FLOAT
        'rotate_angle': 'float64',  # FLOAT
        'bbox_min_long_side': 'float64',  # FLOAT
        'bbox_min_short_side': 'float64',  # FLOAT
        'convexity': 'float64',  # FLOAT
        'concavity': 'float64',  # FLOAT
        'circularity': 'float64',  # FLOAT
        'aspect_ratio': 'float64',  # FLOAT
        'angle': 'float64',  # FLOAT
        'distance_lamina': 'float64',  # FLOAT
        'distance_width': 'float64',  # FLOAT
        'distance_petiole': 'float64',  # FLOAT
        'distance_midvein_span': 'float64',  # FLOAT
        'distance_petiole_span': 'float64',  # FLOAT
        'trace_midvein_distance': 'float64',  # FLOAT
        'trace_petiole_distance': 'float64',  # FLOAT
        'apex_angle': 'float64',  # FLOAT
        'base_angle': 'float64',  # FLOAT
        'base_is_reflex': 'bool',  # BOOLEAN
        'apex_is_reflex': 'bool',  # BOOLEAN

        'filename': 'object',  # TEXT
        'bbox': 'object',  # TEXT
        'bbox_min': 'object',  # TEXT
        'efd_coeffs_features': 'object',  # TEXT 
        'efd_a0': 'object',  # TEXT 
        'efd_c0': 'object',  # TEXT 
        'efd_scale': 'object',  # TEXT
        'efd_angle': 'object',  # TEXT
        'efd_phase': 'object',  # TEXT 
        'efd_area': 'object',  # TEXT 
        'efd_perimeter': 'object',  # TEXT 
        'centroid': 'object',  # TEXT 
        'convex_hull.1': 'object',  # TEXT
        'polygon_closed': 'object',  # TEXT
        'polygon_closed_rotated': 'object',  # TEXT 
        'keypoints': 'object',  # TEXT 
        'tip': 'object',  # TEXT 
        'base': 'object',  # TEXT

        'to_split': 'object',  # TEXT
        'component_type': 'object',  # TEXT
        'component_id': 'object',  # TEXT
        'herb': 'object',  # TEXT
        'gbif_id': 'object',  # TEXT
        'fullname': 'object',  # TEXT
        'genus_species': 'object',  # TEXT
        'family': 'object',  # TEXT
        'genus': 'object',  # TEXT
        'specific_epithet': 'object',  # TEXT
        'cf_error': 'float64',  # TEXT
        'megapixels': 'float64',  # TEXT

        'acceptedNameUsage': 'object',
        'accessRights': 'object',
        'associatedReferences': 'object',
        'associatedSequences': 'object',
        'associatedTaxa': 'object',
        'bibliographicCitation': 'object',
        'collectionCode': 'object',
        'continent': 'object',
        'dataGeneralizations': 'object',
        'datasetID': 'object',
        'datasetName': 'object',
        'dateIdentified': 'object',
        'decimalLatitude': 'object',
        'decimalLongitude': 'object',
        'disposition': 'object',
        'dynamicProperties': 'object',
        'earliestAgeOrLowestStage': 'object',
        'earliestEonOrLowestEonothem': 'object',
        'earliestEraOrLowestErathem': 'object',
        'endDayOfYear': 'object',
        'establishmentMeans': 'object',
        'eventID': 'object',
        'eventRemarks': 'object',
        'eventTime': 'object',
        'fieldNotes': 'object',
        'fieldNumber': 'object',
        'footprintSRS': 'object',
        'footprintWKT': 'object',
        'georeferenceProtocol': 'object',
        'georeferenceRemarks': 'object',
        'georeferenceSources': 'object',
        'georeferenceVerificationStatus': 'object',
        'georeferencedBy': 'object',
        'georeferencedDate': 'object',
        'higherGeography': 'object',
        'identificationID': 'object',
        'identificationReferences': 'object',
        'identificationVerificationStatus': 'object',
        'identifiedBy': 'object',
        'informationWithheld': 'object',
        'infraspecificEpithet': 'object',
        'institutionID': 'object',
        'island': 'object',
        'islandGroup': 'object',
        'language': 'object',
        'latestEonOrHighestEonothem': 'object',
        'latestEpochOrHighestSeries': 'object',
        'level0Gid': 'object',
        'level0Name': 'object',
        'level1Gid': 'object',
        'level1Name': 'object',
        'level2Gid': 'object',
        'level2Name': 'object',
        'level3Gid': 'object',
        'level3Name': 'object',
        'lifeStage': 'object',
        'locationAccordingTo': 'object',
        'locationID': 'object',
        'locationRemarks': 'object',
        'materialSampleID': 'object',
        'month': 'object',
        'municipality': 'object',
        'nomenclaturalCode': 'object',
        'nomenclaturalStatus': 'object',
        'organismID': 'object',
        'organismQuantityType': 'object',
        'otherCatalogNumbers': 'object',
        'ownerInstitutionCode': 'object',
        'parentNameUsage': 'object',
        'parentNameUsageID': 'object',
        'preparations': 'object',
        'previousIdentifications': 'object',
        'projectId': 'object',
        'recordedByID': 'object',
        'reproductiveCondition': 'object',
        'sampleSizeUnit': 'object',
        'samplingEffort': 'object',
        'samplingProtocol': 'object',
        'sex': 'object',
        'startDayOfYear': 'object',
        'taxonConceptID': 'object',
        'taxonID': 'object',
        'taxonRemarks': 'object',
        'type': 'object',
        'typeStatus': 'object',
        'typifiedName': 'object',
        'verbatimCoordinateSystem': 'object',
        'verbatimDepth': 'object',
        'verbatimIdentification': 'object',
        'verbatimLabel': 'object',
        'verbatimLocality': 'object',
        'verbatimSRS': 'object',
        'verbatimTaxonRank': 'object',
        'vernacularName': 'object',
        'waterBody': 'object',
        'year': 'object',

        'elevation': 'object',
        'organismName': 'object',

        'acceptedNameUsageID': 'object',
        'acceptedTaxonKey': 'object',
        'bed': 'object',
        'classKey': 'object',
        'coordinateUncertaintyInMeters': 'object',
        'cultivarEpithet': 'object',
        'depth': 'object',
        'depthAccuracy': 'object',
        'distanceFromCentroidInMeters': 'object',
        'earliestPeriodOrLowestSystem': 'object',
        'elevationAccuracy': 'object',
        'eventType': 'object',
        'familyKey': 'object',
        'footprintSpatialFit': 'object',
        'formation': 'object',
        'genusKey': 'object',
        'geologicalContextID': 'object',
        'group': 'object',
        'higherGeographyID': 'object',
        'highestBiostratigraphicZone': 'object',
        'identifiedByID': 'object',
        'infragenericEpithet': 'object',
        'kingdomKey': 'object',
        'latestAgeOrHighestStage': 'object',
        'latestEraOrHighestErathem': 'object',
        'latestPeriodOrHighestSystem': 'object',
        'lithostratigraphicTerms': 'object',
        'lowestBiostratigraphicZone': 'object',
        'member': 'object',
        'nameAccordingTo': 'object',
        'namePublishedIn': 'object',
        'namePublishedInYear': 'object',
        'orderKey': 'object',
        'originalNameUsage': 'object',
        'originalNameUsageID': 'object',
        'parentEventID': 'object',
        'phylumKey': 'object',
        'scientificNameID': 'object',
        'speciesKey': 'object',
        'subfamily': 'object',
        'subgenus': 'object',
        'subgenusKey': 'object',
        'subtribe': 'object',
        'taxonKey': 'object',
        'tribe': 'object',
        'verticalDatum': 'object',

        'superfamily': 'object',
        'maximumDistanceAboveSurfaceInMeters': 'object',
        'minimumDistanceAboveSurfaceInMeters': 'object',
        'sampleSizeValue': 'object',

        'earliestEpochOrLowestSeries': 'object',
        'nameAccordingToID': 'object',
        'namePublishedInID': 'object',
        'organismRemarks': 'object',

        'associatedOccurrences': 'object',
        'continent': 'object',
        'identificationRemarks': 'object',
        'identifiedBy': 'object',
        'infraspecificEpithet': 'object',
        'recordNumber': 'object',
        'verbatimTaxonRank': 'object',
        'continent': 'object',
        'identificationQualifier': 'object',
        'identificationRemarks': 'object',
        'identifiedBy': 'object',
        'infraspecificEpithet': 'object',
        'level0Gid': 'object',
        'level0Name': 'object',
        'level1Gid': 'object',
        'level1Name': 'object',
        'level2Gid': 'object',
        'level2Name': 'object',
        'level3Gid': 'object',
        'level3Name': 'object',
        'municipality': 'object',
        'occurrenceRemarks': 'object',
        'verbatimElevation': 'object',
        'verbatimTaxonRank': 'object',
    }
    is_test = False

    # Use pandas to get column names
    sample_df = pd.read_csv(file_path, sep="\t", nrows=10000)
    all_columns = sample_df.columns.tolist()

    # Default unspecified columns to 'object'
    complete_dtypes = {col: column_dtypes.get(col, 'object') for col in all_columns}

    # Read the file using Dask
    if is_test:
        df = dd.from_pandas(sample_df, npartitions=1)  # Set npartitions as needed
    else:
        df = dd.read_csv(
            file_path,
            sep="\t",  # Adjust delimiter as needed
            header=0,
            dtype=complete_dtypes,
            assume_missing=True,
            blocksize=f"{block_size}MB",
            on_bad_lines="skip",
            low_memory=False
    )

    # Ensure the 'family' column exists
    if 'family' not in df.columns:
        raise ValueError("The input file does not contain a 'family' column.")
    

    # Group by 'family' and save each group to a separate folder
    # Number of families per batch (adjust based on memory capacity)
    # batch_size = 100000

    # Get unique families
    # families = df['family'].dropna().unique().compute()
    # print(f"\nProcessing {len(families)} families\n")
    # Calculate number of batches
    # Get number of partitions
    num_partitions = df.npartitions
    print(f"\nProcessing {num_partitions} partitions")

    sum_skipped = []
    # Process in batches
    with ProgressBar():
        for i in range(num_partitions):
            print(f"\nProcessing partition {i + 1}/{num_partitions}...")
            
            try:
                # Get the partition as a pandas DataFrame
                partition_df = df.get_partition(i).compute()
                
                # Group the partition by family
                family_groups = partition_df.groupby('family')
                
                # Process each family in the batch
                for family, family_df in family_groups:
                    if family in unique_families:
                        print(f"Processing family: {family}")
                        # Compute only for the current family
                        save_family_csv(output_dir, family, family_df)
                    else:
                        print(f"Skipping family: {family} (not in wishlist)")
            except:
                sum_skipped.append(i)
                print(f"\n\nSKIPPED PARTITION {i}\n\n")
                

def save_family_csv(output_dir, family, family_df):
    """
    Append a group's data to a CSV file in a family-specific folder.

    Args:
        output_dir (str): Base directory for saving family-specific folders.
        family (str): Name of the family (used as folder name).
        family_groups (DataFrameGroupBy): Precomputed group of family rows.
    """

    # Define the family folder and file path
    family_dir = os.path.join(output_dir, family)
    family_file_path = os.path.join(family_dir, 'occurrence.csv')

    # Ensure the family folder exists
    os.makedirs(family_dir, exist_ok=True)

    # Check if the file already exists
    if os.path.exists(family_file_path):
        # Append to the existing file
        existing_df = pd.read_csv(family_file_path, low_memory=False)
        combined_df = pd.concat([existing_df, family_df])
        combined_df.to_csv(family_file_path, index=False)
        print(f"    Appended {len(family_df)} rows to existing file for family '{family}' at {family_file_path}")
    else:
        # Create a new file if it doesn't exist
        family_df.to_csv(family_file_path, index=False)
        print(f"    Created new file for family '{family}' with {len(family_df)} rows at {family_file_path}")
    

if __name__ == '__main__':
    large_file_path = "/media/nas/GBIF_Downloads/Magnoliopsida/occurrence.txt"  # Assuming filename_occ contains the path to the large file
    output_dir = "/media/nas/GBIF_Downloads/Magnoliopsida_By_Family_fast"
    wishlist_csv_path = "/media/nas/GBIF_Downloads/big_tree_names_USE.csv"
    
    save_csv_by_family(large_file_path, output_dir, wishlist_csv_path)