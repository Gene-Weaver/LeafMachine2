import os, json, random, time, csv, dask, platform
import sys
import inspect
import certifi
import ssl
import pandas as pd
import asyncio
from aiohttp import ClientSession, TCPConnector
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
from multiprocessing import Manager, Process

from leafmachine2.machine.general_utils import bcolors, validate_dir
from leafmachine2.downloading.utils_downloads_candidate import ImageCandidate#, PrettyPrint

if platform.system() != "Windows":
    try:
        os.makedirs('/data/tmp', exist_ok=True)
        dask.config.set({'temporary_directory': '/data/tmp'})
    except:
        pass

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

'''
Goal is to simplify the code for test_download_all_images_in_images_csv.py
'''

def load_wishlist_to_tracker(taxonomic_level, wishlist_csv_path=None):
    """
    Imports the 'wishlist' CSV and populates the wishlist_tracker dictionary
    using the 'taxonomic_level' column.
    
    Args:
        wishlist_csv_path (str): The path to the 'wishlist' CSV file.
        
    Returns:
        wishlist_tracker (manager.dict()): A shared dictionary with wishlist taxa categorized by taxonomic level.
    """
    # Initialize a Manager dictionary for shared memory
    manager = Manager()
    wishlist_tracker = manager.list()

    try:
        # Load the wishlist CSV into a DataFrame
        wishlist_df = pd.read_csv(wishlist_csv_path, sep=",", header=0, low_memory=False, dtype=str, on_bad_lines='skip')

        # Ensure the CSV contains the required columns
        if taxonomic_level not in wishlist_df.columns:
            raise ValueError(f"The CSV must contain {taxonomic_level} column.")

        # Process each row in the DataFrame
        wishlist_tracker = wishlist_df[taxonomic_level].tolist()

        print(f"Wishlist tracker populated with {len(wishlist_tracker)} taxonomic levels.")
        return wishlist_tracker

    except Exception as e:
        print(f"Error loading wishlist CSV: {e}")
        return None


def run_download_parallel(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, filename_csv_family_counts_core="set"):
    # pp = PrettyPrint()
    pp = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_main(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp, filename_csv_family_counts_core))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def run_main(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp, filename_csv_family_counts_core):
    # semaphore_scraperapi = asyncio.Semaphore(5)
    await download_all_images_in_images_csv_selenium(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp, filename_csv_family_counts_core)

async def download_all_images_in_images_csv_selenium(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp, filename_csv_family_counts_core):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']
    min_occ_cutoff = cfg['min_occ_cutoff']

    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)

    if cfg['is_custom_file']:
        ImageBatchProcessor.download_from_custom_file(cfg)
        return

    # Load occurrence and image data from the Darwin Core file
    occ_df, images_df = ImageBatchProcessor.read_DWC_file(cfg)
    print("Debug: Initial images_df type:", type(images_df))

    # If no occurrence data is available, exit the function
    if occ_df is None:
        return
    
    '''# Display debug information about the sampled images DataFrame (optional sampling)
    # images_df = images_df.sample(frac=1, random_state=2023).reset_index(drop=True)
    print("Debug: After sampling, images_df type:", type(images_df))

    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())
    print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
    print(occ_df.head())

    # Filter out banned URLs if specified in the configuration
    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]
    print("Debug: After filtering banned herbs, images_df type:", type(images_df))

    # Identify relevant ID columns in both dataframes
    occ_id_column = next((col for col in ['id', 'gbifID'] if col in occ_df.columns), None)
    images_id_column = next((col for col in ['coreid', 'gbifID'] if col in images_df.columns), None)

    # Raise an error if required ID columns are missing
    if not (occ_id_column and images_id_column):
        raise ValueError("Required ID columns not found in dataframes.")

    # Ensure ID columns are strings and filter images DataFrame to include only matching IDs
    occ_df[occ_id_column] = occ_df[occ_id_column].astype(str)
    images_df[images_id_column] = images_df[images_id_column].astype(str)
    images_df = images_df[images_df[images_id_column].isin(occ_df[occ_id_column])]
    print("Debug: After filtering by ID columns, images_df type:", type(images_df))

    

    # Ensure fullname exists in occ_df
    # Create a 'fullname_index' column in occurrences DataFrame, combining taxonomic hierarchy
    occ_df['fullname_index'] = occ_df.apply(lambda row: f"{row['family']}_{row['genus']}_{row['specificEpithet'] if pd.notna(row['specificEpithet']) else ''}", axis=1)


    # Create a dictionary of counts for each fullname_index
    fullname_index_counts = occ_df['fullname_index'].value_counts().to_dict()
    # Explicitly sort the dictionary by values (counts) in descending order (optional)
    fullname_index_counts_sorted = dict(sorted(fullname_index_counts.items(), key=lambda item: item[1], reverse=True))
    print(f"Fullname Index Counts (sorted): {fullname_index_counts_sorted}")

    # Save the sorted 
    df_family_counts = pd.DataFrame(fullname_index_counts_sorted.items(), columns=['fullname_index', 'count'])
    # Generate the filename and path
    filename_csv_family_counts = '_'.join([cfg['filename_csv_family_counts_stem'], filename_csv_family_counts_core]) + ".csv"
    path_csv_family_counts = os.path.join(cfg['dir_home'], filename_csv_family_counts)
    # Save the DataFrame to a CSV file
    df_family_counts.to_csv(path_csv_family_counts, index=False)
    print(f"Sorted family counts saved to: {path_csv_family_counts}")
    '''
    # Check if Dask or pandas DataFrame and handle accordingly
    if isinstance(occ_df, dd.DataFrame):
        print("Processing occ_df and images_df as Dask DataFrames")
        
        # Filter out banned URLs in Dask
        if cfg['ignore_banned_herb']:
            for banned_url in cfg['banned_url_stems']:
                images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]

        # Identify relevant ID columns in both dataframes
        occ_id_column = next((col for col in ['id', 'gbifID'] if col in occ_df.columns), None)
        images_id_column = next((col for col in ['coreid', 'gbifID'] if col in images_df.columns), None)

        if not (occ_id_column and images_id_column):
            raise ValueError("Required ID columns not found in dataframes.")

        # Compute the unique IDs from occ_df into a list
        occ_ids = occ_df[occ_id_column].drop_duplicates().compute()

        # Convert to a list for compatibility with `isin`
        occ_ids_list = occ_ids.tolist()

        # Filter images_df based on matching IDs
        images_df = images_df[images_df[images_id_column].isin(occ_ids_list)]

        # Create fullname_index column
        occ_df['fullname_index'] = occ_df.map_partitions(
            lambda df: df.apply(
                lambda row: f"{row['family']}_{row['genus']}_{row['specificEpithet'] if pd.notna(row['specificEpithet']) else ''}", 
                axis=1
            )
        )

        # If a wishlist tracker is provided, filter occurrences based on the wishlist
        if wishlist_tracker:
            print("Filtering based on wishlist tracker...")
            
            # Compute wishlist tracker to a list if it’s a Dask object
            if isinstance(wishlist_tracker, dd.Series):
                wishlist_tracker = wishlist_tracker.compute().tolist()
            elif isinstance(wishlist_tracker, pd.Series):
                wishlist_tracker = wishlist_tracker.tolist()

            # Apply filtering
            print(f"Before filtering: 'fullname_index' count = {occ_df['fullname_index'].count().compute()}")
            occ_df = occ_df[occ_df['fullname_index'].isin(wishlist_tracker)]
            print(f"After filtering: 'fullname_index' count = {occ_df['fullname_index'].count().compute()}")

        # Calculate counts using Dask and compute
        fullname_index_counts = occ_df['fullname_index'].value_counts().compute()

        print(f"Before cutoff filtering: 'fullname_index' count = {occ_df['fullname_index'].count().compute()}")

        # Filter fullname_index values based on min_occ_cutoff
        valid_fullnames = fullname_index_counts[fullname_index_counts >= min_occ_cutoff].index.tolist()
        # Filter occ_df based on valid fullname_index
        occ_df = occ_df[occ_df['fullname_index'].isin(valid_fullnames)]
        images_df = images_df[images_df['gbifID'].isin(occ_ids_list)]  # Use the filtered `occ_ids_list`
        fullname_index_counts_sorted = fullname_index_counts.sort_values(ascending=False)

    else:
        print("Processing occ_df and images_df as pandas DataFrames")

        # Process pandas DataFrames (existing logic)
        if cfg['ignore_banned_herb']:
            for banned_url in cfg['banned_url_stems']:
                images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]

        # Identify relevant ID columns in both dataframes
        occ_id_column = next((col for col in ['id', 'gbifID'] if col in occ_df.columns), None)
        images_id_column = next((col for col in ['coreid', 'gbifID'] if col in images_df.columns), None)

        if not (occ_id_column and images_id_column):
            raise ValueError("Required ID columns not found in dataframes.")

        # Filter images_df based on matching IDs
        occ_df[occ_id_column] = occ_df[occ_id_column].astype(str)
        images_df[images_id_column] = images_df[images_id_column].astype(str)
        images_df = images_df[images_id_column].isin(occ_df[occ_id_column])

        # Create fullname_index column
        occ_df['fullname_index'] = occ_df.apply(
            lambda row: f"{row['family']}_{row['genus']}_{row['specificEpithet'] if pd.notna(row['specificEpithet']) else ''}",
            axis=1
        )

        # If a wishlist tracker is provided, filter occurrences based on the wishlist
        if wishlist_tracker:
            # Print the length of the 'fullname_index' column before filtering
            print(f"Before filtering: 'fullname_index' length = {len(occ_df['fullname_index'])}")

            # Filter occ_df based on wishlist_tracker['fullname'] column
            occ_df = occ_df[occ_df['fullname_index'].isin(wishlist_tracker)]
            
            print(f"After filtering: 'fullname_index' length = {len(occ_df['fullname_index'])}")

        # Calculate counts
        fullname_index_counts = occ_df['fullname_index'].value_counts()

        # Filter fullname_index values based on min_occ_cutoff
        valid_fullnames = fullname_index_counts[fullname_index_counts >= min_occ_cutoff].index.tolist()
        # Filter occ_df based on valid fullname_index
        occ_df = occ_df[occ_df['fullname_index'].isin(valid_fullnames)]
        images_df = images_df[images_df['gbifID'].isin(occ_df[occ_id_column])]

        fullname_index_counts_sorted = fullname_index_counts.sort_values(ascending=False)

        

    # Save the sorted counts to a CSV file
    df_family_counts = pd.DataFrame(
        fullname_index_counts_sorted.items(), columns=['fullname_index', 'count']
    )
    filename_csv_family_counts = '_'.join([cfg['filename_csv_family_counts_stem'], filename_csv_family_counts_core]) + ".csv"
    path_csv_family_counts = os.path.join(cfg['dir_home'], filename_csv_family_counts)
    df_family_counts.to_csv(path_csv_family_counts, index=False)
    print(f"Sorted family counts saved to: {path_csv_family_counts}")


    if isinstance(occ_df, dd.DataFrame):
        print(f"    Computing occ_df...")
        occ_df = occ_df.compute()
        print(len(occ_df['fullname_index']))
    if isinstance(images_df, dd.DataFrame):
        print(f"    Computing images_df...")
        images_df = images_df.compute()
        print(len(images_df))
    

    


    # Shuffle the DataFrame with a random seed
    # Shuffle the occurrences DataFrame for random processing order
    occ_df = occ_df.sample(frac=1, random_state=2023).reset_index(drop=True)

    # Sort occurrences by the frequency of 'fullname_index' to prioritize less frequent taxa
    # fullname_counts = occ_df['fullname_index'].value_counts().to_dict()
    # occ_df['fullname_frequency'] = occ_df['fullname_index'].map(fullname_counts)
    # occ_df = occ_df.sort_values(by='fullname_frequency', ascending=True).reset_index(drop=True)
    # occ_df = occ_df.drop(columns=['fullname_frequency'])


    print(f"{bcolors.BOLD}Number of images in images file: {images_df.shape[0]}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Number of occurrence to search through: {occ_df.shape[0]}{bcolors.ENDC}")

    # Initialize an image processor with the filtered data and trackers
    processor = ImageBatchProcessor(cfg, images_df, occ_df, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp)
    processor.init_selenium_drivers()

    try:
        await processor.process_batch()
    finally:
        await processor.finalize()

def detect_delimiter(file_path, fallback_delimiters=['\t', ',']):
    # # Check if the file exists
    # if not os.path.isfile(file_path):
    #     raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # # Open the file and read the sample
    # with open(file_path, 'r') as f:
    #     sample = f.read(1024)  # Read the first 1024 bytes
    
    # # Try to detect delimiter using csv.Sniffer
    # try:
    #     detected_delimiter = csv.Sniffer().sniff(sample).delimiter
    # except csv.Error:
    # Fallback to manually testing common delimiters
    if '.txt' in file_path:
        return '\t'
    elif '.csv' in file_path:
        return ','
    else:
    # Raise an error if no delimiter is found
        raise ValueError("Could not determine delimiter from the file or fallback options.")
    
    # return detected_delimiter


class ImageBatchProcessor:
    def __init__(self, cfg, images_df, occ_df, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp ):
        self.cfg = cfg
        self.images_df = images_df
        self.occ_df = occ_df
        self.results = []
        self.num_workers = self.cfg.get('n_threads', 8)
        self.num_drivers = self.cfg.get('num_drivers', 8)
        self.failure_log = {}
        # self.semaphore_scraperapi = semaphore_scraperap

        self.failure_queue = asyncio.Queue()
        self.proxy_queue = asyncio.Queue()
        self.selenium_pool = []
        self.proxy_pool = []
        self.driver_queue = asyncio.Queue()

        self.n_queue = 0
        self.total_rows = len(images_df)
        self.processed_rows = 0
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(self.num_drivers)

        self.wishlist_tracker = wishlist_tracker
        self.download_tracker = download_tracker
        self.completed_tracker = completed_tracker
        self.banned_url_tracker = banned_url_tracker
        self.banned_url_counts_tracker = banned_url_counts_tracker
        
        self.pp = pp
        self.shutdown_flag = asyncio.Event()  # Add the shutdown flag

    def get_driver_with_random_user_agent(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        ]

        options = Options()
        user_agent = random.choice(user_agents)
        options.add_argument(f"user-agent={user_agent}")
        options.headless = True

        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        return driver

    def init_selenium_drivers(self):
        for index in range(self.num_drivers):
            driver = self.get_driver_with_random_user_agent()
            self.selenium_pool.append(driver)
            self.driver_queue.put_nowait((index, driver))


    def close_selenium_drivers(self):
        while not self.driver_queue.empty():
            _, driver = self.driver_queue.get_nowait()
            driver.quit()
        self.selenium_pool = []


    def find_images_for_gbifID(self, gbif_id):
        gbif_id = str(gbif_id)
        return self.images_df[self.images_df['gbifID'].astype(str) == gbif_id]

    async def fetch_image(self, session, occ_row, semaphore):
        try:
            id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in occ_row), None)
            if id_column is None:
                raise KeyError("No valid ID column found in occ_row.")
            
            gbif_id = occ_row[id_column]
            image_rows = self.find_images_for_gbifID(gbif_id)

            if image_rows.shape[0] == 0:
                return None
            
            if occ_row['fullname_index'] in self.completed_tracker:
                return None

            for _, image_row in image_rows.iterrows():
                gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['identifier']

                if pd.isna(gbif_url):
                    try: # try the references column
                        gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['references']
                        if pd.isna(gbif_url):
                            return None
                    except:
                        return None



                image_candidate = ImageCandidate(self.cfg, occ_row, image_row, gbif_url, self.failure_log, self.download_tracker, self.completed_tracker, self.banned_url_tracker, self.banned_url_counts_tracker)
                async with semaphore:
                    print(f"Working on image: {image_row[id_column]}")

                    # Non-blocking download attempt
                    await image_candidate.download_image(session, self.n_queue, logging_enabled=False)

                    if image_candidate.download_success == 'skip':
                        print(f"Skipping 404 error for image: {image_row[id_column]}")
                    elif image_candidate.download_success == False:
                        # Offload failure to failure queue for further handling by Selenium/ScraperAPI
                        await self.failure_queue.put(image_candidate)
                        self.n_queue = self.failure_queue.qsize()
                        print(f"{bcolors.CBLACKBG}QUEUE [{self.n_queue}]{bcolors.ENDC}")

                async with self.lock:
                    self.processed_rows += 1

            return pd.DataFrame(occ_row) if image_candidate.download_success == True else None
        except Exception as e:
            print(f"Error in fetch_image: {e}")
            return None

    async def fetch_image_with_proxy(self, session, image_candidate):
        try:
            await image_candidate.download_image_with_proxy(session)
            print(f"{bcolors.CGREEN}Successfully processed proxy image: {image_candidate.fullname}{bcolors.ENDC}")
        except Exception as e:
            print(f"{bcolors.CRED}Error in fetch_image_with_proxy: {e}{bcolors.ENDC}")


    def remove_fullnames_with_all_images(self, fullname):
        if fullname in self.completed_tracker:
            return
        if fullname in self.download_tracker and self.download_tracker[fullname] >= self.cfg['n_to_download']:
            print(f"B = {self.occ_df.shape[0]}")
            self.occ_df = self.occ_df[self.occ_df['fullname_index'] != fullname]
            print(f"A = {self.occ_df.shape[0]}")
            self.completed_tracker.append(fullname)
            for fn in self.completed_tracker:
                # self.pp.add_completed(fn)
                print(f"{bcolors.CGREYBG}COMPLETED [{fn}]{bcolors.ENDC}")


    async def producer(self, queue):
        print(f"Producer: Starting to iterate over occ_df")
        for _, occ_row in self.occ_df.iterrows():
            await queue.put(occ_row)
        for _ in range(self.num_workers):
            await queue.put(None)  # Stop signal for consumers
        print(f"Producer: Finished adding items to the queue")


    async def consumer(self, session, queue, semaphore):
        print(f"Consumer: Waiting for items")
        while True:
            image_row = await queue.get()
            if image_row is None:
                queue.task_done()  # Ensure task_done is called for the stop signal
                print(f"Consumer received shutdown signal.")
                break
            await self.fetch_image(session, image_row, semaphore)
            queue.task_done()
        print(f"Consumer: Finished processing items")

    def count_active_drivers(self):
        return sum(1 for d in self.selenium_pool if d.service.process is not None and d.service.process.poll() is None)

    async def failure_consumer(self, index, driver):
        while True:
            try:
                if self.failure_queue.empty() and self.shutdown_flag.is_set():
                    print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
                    print(f"{bcolors.CGREENBG}Failure Consumer {index}: received shutdown signal.{bcolors.ENDC}")
                    print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
                    break

                try:
                    image_candidate = await asyncio.wait_for(self.failure_queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    continue

                if image_candidate is not None:
                    await self.process_with_driver(image_candidate, driver, index)
                    self.failure_queue.task_done()

                    # After retrying with Selenium, check if a proxy retry is necessary
                    if image_candidate.download_success == 'proxy':
                        await self.proxy_queue.put(image_candidate)
                        print(f"{bcolors.HEADER}Added to proxy queue for image: {image_candidate.fullname}{bcolors.ENDC}")


                # Check if driver is still active
                # if driver.service.process is None or driver.service.process.poll() is not None:
                #     driver.quit()
                #     self.selenium_pool[index] = self.get_driver_with_random_user_agent()
                #     driver = self.selenium_pool[index]
                #     self.driver_queue.put_nowait((index, driver))

                # Ensure the total number of drivers is maintained
                # async with self.lock:
                active_drivers_count = self.count_active_drivers()
                while active_drivers_count < self.num_drivers:
                    print(f"{bcolors.CGREENBG}Driver is no longer active. [{active_drivers_count}] to [{active_drivers_count+1}]Reinitializing...{bcolors.ENDC}")
                    new_driver = self.get_driver_with_random_user_agent()
                    self.selenium_pool.append(new_driver)
                    self.driver_queue.put_nowait((len(self.selenium_pool) - 1, new_driver))
                    active_drivers_count += 1

                    # if image_candidate.download_success == 'proxy':
                    #     await self.proxy_queue.put(image_candidate)
                    #     n_queue_proxy = self.proxy_queue.qsize()
                    #     print(f"{bcolors.HEADER}Added to proxy queue for image: {image_candidate.fullname}. PROXY QUEUE REMAINING {n_queue_proxy}{bcolors.ENDC}")
            except Exception as e:
                print(f"Failure Consumer {index} encountered an error: {e}")
                continue  # Ensure the consumer keeps running even if an error occurs

        print(f"Failure Consumer {index}: Finished processing items")


    async def process_batch(self):
        # Create an SSL context for secure connections
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(ssl=ssl_context)

        # Define the number of concurrent tasks allowed by the semaphore
        n_semaphore = self.num_workers
        if n_semaphore <= 2:
            n_semaphore = self.num_workers  # Ensure a minimum number of workers
        semaphore = asyncio.Semaphore(n_semaphore)

        # Create a queue to handle tasks for producers and consumers
        queue = asyncio.Queue(maxsize=1000)

        # Use an asynchronous HTTP session for making network requests
        async with ClientSession(connector=connector) as session:
            # Initialize producer tasks to add items to the queue
            producers = [asyncio.create_task(self.producer(queue))]
            
            # Initialize consumer tasks to process items from the queue
            consumers = [asyncio.create_task(self.consumer(session, queue, semaphore)) for _ in range(self.num_workers)]
            
            # Initialize failure consumer tasks to handle failures using Selenium
            failure_consumers = [asyncio.create_task(self.failure_consumer(index, driver)) for index, driver in enumerate(self.selenium_pool)]

            # Wait for all producer tasks to complete
            await asyncio.gather(*producers)
            
            # Ensure all tasks in the queue are processed before continuing
            await queue.join()

            # Send stop signals to all consumer tasks
            for _ in range(self.num_workers):
                await queue.put(None)  # Add a `None` item as a shutdown signal for each consumer
            
            # Wait for all consumer tasks to complete
            await asyncio.gather(*consumers)
            print(f"{bcolors.CGREENBG}All consumers have finished processing.{bcolors.ENDC}")

            # Set a shutdown flag to indicate the failure processing loop should stop
            self.shutdown_flag.set()

            # Ensure all failure tasks are processed before continuing
            await self.failure_queue.join()

            # Send stop signals to failure consumer tasks
            for _ in range(self.count_active_drivers()):
                await self.failure_queue.put(None)  # Add a `None` item as a shutdown signal for each failure consumer

            # Wait for all failure consumer tasks to complete
            await asyncio.gather(*failure_consumers)
            print(f"{bcolors.CGREENBG}All failure consumers have shut down.{bcolors.ENDC}")

            # Perform final cleanup actions
            await self.finalize()



    async def process_with_driver(self, image_candidate, driver, index):
        active_drivers_count = sum(1 for d in self.selenium_pool if d.service.process is not None and d.service.process.poll() is None)

        print(f"{bcolors.CWHITEBG}[Active drivers {active_drivers_count}]{bcolors.ENDC}")

        # await asyncio.get_event_loop().run_in_executor(
        #     self.executor, self.sync_wrapper, image_candidate, driver, index, self.n_queue
        # )
        # await asyncio.to_thread(self.sync_wrapper, image_candidate, driver, index, self.n_queue)
        await asyncio.to_thread(self.sync_wrapper, image_candidate, driver, index, self.n_queue)


        # await image_candidate.download_image_with_selenium(driver, index, self.n_queue, self.semaphore_scraperapi)
        n_queue = self.failure_queue.qsize()
        n_queue_proxy = self.proxy_queue.qsize()

        if image_candidate.download_success == 'skip':
            print(f"Skipping image processing for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")
        elif not image_candidate.download_success:
            print(f"Failed processing with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")
        else:
            print(f"Successfully processed with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")


    @staticmethod
    def sync_wrapper(image_candidate, driver, index, n_queue):
        # Ensure a new event loop for each thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


    async def finalize(self):
        self.close_selenium_drivers()
        self.executor.shutdown(wait=True)
        print(f"{bcolors.CGREENBG}All tasks are completed and selenium drivers are closed.{bcolors.ENDC}")



    def find_gbifID(self, gbif_id):
        file_to_search = self.occ_df
        if 'gbifID' in file_to_search.columns:
            row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)), :]
        elif 'id' in file_to_search.columns:
            row_found = file_to_search.loc[file_to_search['id'].astype(str).str.match(str(gbif_id)), :]
        elif 'coreid' in file_to_search.columns:
            row_found = file_to_search.loc[file_to_search['coreid'].astype(str).str.match(str(gbif_id)), :]
        else:
            raise KeyError("Neither 'gbifID' nor 'id' found in the column names for the Occurrences file")

        if row_found.empty:
            # print(f"gbif_id: {gbif_id} not found in occurrences file")
            return None
        else:
            # print(f"gbif_id: {gbif_id} successfully found in occurrences file")
            return row_found

    def save_results(self, combined_results: pd.DataFrame):
        path_csv_combined = os.path.join(self.cfg['dir_destination_csv'], self.cfg['filename_combined'])
        try:
            combined_results.to_csv(path_csv_combined, mode='a', header=False, index=False)
            print(f'{bcolors.OKGREEN}Successfully saved combined results to {path_csv_combined}{bcolors.ENDC}')
        except Exception as e:
            print(f'{bcolors.FAIL}Failed to save combined results: {e}{bcolors.ENDC}')

    def save_failure_log(self):
        path_error_log = os.path.join(self.cfg['dir_destination_csv'], 'error_log.json')
        try:
            with open(path_error_log, 'w') as f:
                json.dump(self.failure_log, f, indent=4)
            print(f'Successfully saved failure log to {path_error_log}')
        except Exception as e:
            print(f'Failed to save failure log: {e}')

    @staticmethod
    def read_custom_file(cfg):
        dir_home = cfg['dir_home']
        filename_img = cfg['filename_img']
        images_df = ImageBatchProcessor.ingest_DWC(filename_img, dir_home)
        return images_df

    @staticmethod
    def download_from_custom_file(cfg):
        images_df = ImageBatchProcessor.read_custom_file(cfg)

        col_url = cfg.get('col_url', 'identifier')
        col_name = cfg['col_name']

        # Report summary
        print(f"Beginning of images file:")
        print(images_df.head())

        # Ignore problematic Herbaria
        if cfg['ignore_banned_herb']:
            for banned_url in cfg['banned_url_stems']:
                images_df = images_df[~images_df[col_url].str.contains(banned_url, na=False)]

        # Report summary
        n_imgs = images_df.shape[0]
        print(f"Number of images in images file: {n_imgs}")

        processor = ImageBatchProcessor(cfg, images_df)
        asyncio.run(processor.process_batch())

    @staticmethod
    def read_DWC_file(cfg):
        dir_home = cfg['dir_home']
        filename_occ = cfg['filename_occ']
        filename_img = cfg['filename_img']
        project_multimedia_file = cfg['project_multimedia_file'] # If this is not None, then use it instead of the dynamically generated img path


        try:
            use_large = cfg['use_large_file_size_methods']
        except:
            use_large = False

        if use_large:
            occ_df = ImageBatchProcessor.ingest_DWC_large_files(filename_occ, dir_home, None, do_compute=False)
            images_df = ImageBatchProcessor.ingest_DWC_large_files(filename_img, dir_home, project_multimedia_file, do_compute=False)
        else:
            occ_df = ImageBatchProcessor.ingest_DWC(filename_occ, dir_home, None)
            images_df = ImageBatchProcessor.ingest_DWC(filename_img, dir_home, project_multimedia_file)
            
        
        return occ_df, images_df

    @staticmethod
    def ingest_DWC(DWC_csv_or_txt_file, dir_home, project_multimedia_file):

        if project_multimedia_file:
            file_path = project_multimedia_file
        else:
            file_path = os.path.join(dir_home, DWC_csv_or_txt_file)
        file_extension = DWC_csv_or_txt_file.split('.')[-1]


        try:
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

        return df
    
    @staticmethod
    def ingest_DWC_large_files(file_path, dir_home, project_multimedia_file, do_compute=True, block_size=512):
        # Method to ingest large DWC (Darwin Core) CSV or TXT files using Dask to handle larger-than-memory files.
        # This function supports .txt or .csv files with varying delimiters.
        if project_multimedia_file:
            file_path = project_multimedia_file
            file_name = os.path.basename(file_path)
        else:
            file_path = os.path.join(dir_home, file_path)
            file_name = os.path.basename(file_path)
        file_extension = file_path.split('.')[-1]

        # Define the dtype for each column
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
            'identificationQualifier': 'object',
            'identificationRemarks': 'object',
            'occurrenceRemarks': 'object',
            'recordNumber': 'object',
            'verbatimElevation': 'object',
            'associatedOccurrences': 'object',

            'created': 'object',
            'publisher': 'object',
            'title': 'object',
        }
        # Check if the file name contains "occurrences" or "multimedia"
        # if "occurrences" in file_name.lower() or "occurrence" in file_name.lower() or "multimedia" in file_name.lower():
            # Read the file without specifying dtypes to infer the column names
            # df_tmp = dd.read_csv(file_path, sep=None, header=0, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True)
            # df_tmp = dd.read_csv(file_path, sep=None, header=0, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True, nrows=5)
            # # Set all columns to dtype object (equivalent to strings in pandas/dask)
            # column_names = df_tmp.columns.tolist()
            # # column_dtypes = {col: 'object' for col in df_tmp.columns}
            # column_dtypes = {col: 'object' for col in column_names}

            # Read a small portion of the file to get the column names
            # df_tmp = dd.read_csv(file_path, sep="\t", dtype='object',header=0, low_memory=False, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True).head(n=5)
            # column_names = df_tmp.columns.tolist()
            
            # # Set all columns to dtype object (equivalent to strings in pandas/dask)
            # column_dtypes = {col: 'object' for col in column_names}
        # Use pandas to get column names
        try:
            delimiter = detect_delimiter(file_path)
            print(f"Detected delimiter: {delimiter}")
        except Exception as e:
            print(f"Error: {e}")

        sample_df = pd.read_csv(file_path, sep=delimiter, nrows=10, low_memory=False)
        all_columns = sample_df.columns.to_list()

        # Default unspecified columns to 'object'
        complete_dtypes = {col: column_dtypes.get(col, 'object') for col in all_columns}

        try:
            if "occurrences" in file_name.lower() or "occurrence" in file_name.lower() or "multimedia" in file_name.lower():
                if file_extension == 'txt':
                    # Reading a .txt file with tab-delimited data
                    df = dd.read_csv(file_path, sep="\t", header=0, dtype=complete_dtypes, assume_missing=True, 
                                    blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                elif file_extension == 'csv':
                    # Reading a .csv file (comma-separated)
                    df = dd.read_csv(file_path, sep=",", header=0, dtype=complete_dtypes, assume_missing=True, 
                                    blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                else:
                    # Handle other cases (e.g., pipe-separated or semicolon-separated)
                    try:
                        df = dd.read_csv(file_path, sep="|", header=0, dtype=complete_dtypes, assume_missing=True, 
                                        blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                    except Exception:
                        try:
                            df = dd.read_csv(file_path, sep=";", header=0, dtype=complete_dtypes, assume_missing=True, 
                                            blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                        except Exception as e:
                            print(f"Error reading file with different delimiter: {e}")
                            return None
            else:
                print(f"DWC file {file_path} is not '.txt' or '.csv' and was not opened")
                return None
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None

        df = df.astype(complete_dtypes)
        if do_compute:
            return df.compute()
        else:
            return df
