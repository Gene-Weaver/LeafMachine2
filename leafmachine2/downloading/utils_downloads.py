import os, json, random, time
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

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)



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
    

# def run_download_parallel(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker):
#     # pp = PrettyPrint()
#     pp = None
#     asyncio.run(run_main(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp))
def run_download_parallel(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker):
    # pp = PrettyPrint()
    pp = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_main(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

async def run_main(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp):
    # semaphore_scraperapi = asyncio.Semaphore(5)
    await download_all_images_in_images_csv_selenium(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp)

async def download_all_images_in_images_csv_selenium(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']

    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)

    if cfg['is_custom_file']:
        ImageBatchProcessor.download_from_custom_file(cfg)
        return

    occ_df, images_df = ImageBatchProcessor.read_DWC_file(cfg)
    print("Debug: Initial images_df type:", type(images_df))

    if occ_df is None:
        return
    
    # images_df = images_df.sample(frac=1, random_state=2023).reset_index(drop=True)
    print("Debug: After sampling, images_df type:", type(images_df))

    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())
    print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
    print(occ_df.head())

    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]
    print("Debug: After filtering banned herbs, images_df type:", type(images_df))

    occ_id_column = next((col for col in ['id', 'gbifID'] if col in occ_df.columns), None)
    images_id_column = next((col for col in ['coreid', 'gbifID'] if col in images_df.columns), None)

    if not (occ_id_column and images_id_column):
        raise ValueError("Required ID columns not found in dataframes.")

    occ_df[occ_id_column] = occ_df[occ_id_column].astype(str)
    images_df[images_id_column] = images_df[images_id_column].astype(str)
    images_df = images_df[images_df[images_id_column].isin(occ_df[occ_id_column])]
    print("Debug: After filtering by ID columns, images_df type:", type(images_df))

    

    # Ensure fullname exists in occ_df
    occ_df['fullname_index'] = occ_df.apply(lambda row: f"{row['family']}_{row['genus']}_{row['specificEpithet'] if pd.notna(row['specificEpithet']) else ''}", axis=1)
    
    if wishlist_tracker:
        # Print the length of the 'fullname_index' column before filtering
        print(f"Before filtering: 'fullname_index' length = {len(occ_df['fullname_index'])}")

        # Filter occ_df based on wishlist_tracker['fullname'] column
        occ_df = occ_df[occ_df['fullname_index'].isin(wishlist_tracker)]
        
        print(f"After filtering: 'fullname_index' length = {len(occ_df['fullname_index'])}")


    # Shuffle the DataFrame with a random seed
    occ_df = occ_df.sample(frac=1, random_state=2023).reset_index(drop=True)

    # Sort by the frequency of fullname_index
    fullname_counts = occ_df['fullname_index'].value_counts().to_dict()
    occ_df['fullname_frequency'] = occ_df['fullname_index'].map(fullname_counts)
    occ_df = occ_df.sort_values(by='fullname_frequency', ascending=True).reset_index(drop=True)
    occ_df = occ_df.drop(columns=['fullname_frequency'])


    print(f"{bcolors.BOLD}Number of images in images file: {images_df.shape[0]}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Number of occurrence to search through: {occ_df.shape[0]}{bcolors.ENDC}")

    processor = ImageBatchProcessor(cfg, images_df, occ_df, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp)
    processor.init_selenium_drivers()

    try:
        await processor.process_batch()
    finally:
        await processor.finalize()



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

    # async def proxy_consumer(self, proxy_queue, semaphore_proxy):
    #     async with ClientSession() as session:
    #         print(f"Proxy Consumer: Waiting for items")
    #         while True:
    #             try:
    #                 image_candidate = await proxy_queue.get()
    #                 if image_candidate is None:
    #                     proxy_queue.task_done()  # Ensure task_done is called for the stop signal
    #                     print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
    #                     print(f"{bcolors.CGREENBG}Proxy Consumer: received shutdown signal.{bcolors.ENDC}")
    #                     print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
    #                     break

    #                 print(f"{bcolors.CYELLOW}Processing proxy item: {image_candidate.fullname}{bcolors.ENDC}")

    #                 async with semaphore_proxy:
    #                     await self.fetch_image_with_proxy(session, image_candidate)
                    
    #                 proxy_queue.task_done()
    #             except Exception as e:
    #                 print(f"Proxy Consumer encountered an error: {e}")
    #                 continue  # Ensure the consumer keeps running even if an error occurs
    #         print(f"Proxy Consumer: Finished processing items")
    # async def proxy_consumer(self, session, proxy_queue, semaphore_proxy):
    #     # async with ClientSession() as session:
    #     while True:
    #         image_candidate = await proxy_queue.get()
    #         if image_candidate is None:
    #             proxy_queue.task_done()
    #             break

    #         async with semaphore_proxy:
    #             await self.fetch_image_with_proxy(session, image_candidate)
    #         proxy_queue.task_done()

    async def process_batch(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(ssl=ssl_context)

        n_semaphore = self.num_workers
        if n_semaphore <= 2:
            n_semaphore = self.num_workers
        semaphore = asyncio.Semaphore(n_semaphore)

        # semaphore_proxy = asyncio.Semaphore(5)  # Limit the number of concurrent proxy consumers
        queue = asyncio.Queue(maxsize=1000)
        # proxy_queue = asyncio.Queue(maxsize=1000)  # Queue for proxy consumers

        async with ClientSession(connector=connector) as session:
            producers = [asyncio.create_task(self.producer(queue))]
            consumers = [asyncio.create_task(self.consumer(session, queue, semaphore)) for _ in range(self.num_workers)]
            failure_consumers = [asyncio.create_task(self.failure_consumer(index, driver)) for index, driver in enumerate(self.selenium_pool)]
            # proxy_consumers = [asyncio.create_task(self.proxy_consumer(session, proxy_queue, semaphore_proxy)) for _ in range(5)]

            await asyncio.gather(*producers)
            await queue.join()

            for _ in range(self.num_workers):
                await queue.put(None)  # Stop signal for consumers
            await asyncio.gather(*consumers)
            print(f"{bcolors.CGREENBG}All consumers have finished processing.{bcolors.ENDC}")

            self.shutdown_flag.set()

            await self.failure_queue.join()  # Ensure all failure tasks are processed
            # Add shutdown signals for failure consumers
            for _ in range(self.count_active_drivers()):
                await self.failure_queue.put(None)

            await asyncio.gather(*failure_consumers)
            print(f"{bcolors.CGREENBG}All failure consumers have shut down.{bcolors.ENDC}")


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


    # @staticmethod
    # def sync_wrapper(image_candidate, driver, index, n_queue):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     retries = 1
    #     backoff = 1
    #     for attempt in range(retries):
    #         # result = loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
    #         loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
    #         # if result:
    #             # break
    #         # time.sleep(backoff * (2 ** attempt))
    #     loop.close()
    #     return #result

    # @staticmethod # 10/12/24
    # def sync_wrapper(image_candidate, driver, index, n_queue):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         retries = 1
    #         for attempt in range(retries):
    #             try:
    #                 loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
    #                 if image_candidate.download_success == True and image_candidate.download_success not in ['skip', 'proxy']:
    #                     return  # Exit after the first successful attempt
    #             except Exception as e:
    #                 print(f"Attempt {attempt + 1} failed with error: {e}")
    #     finally:
    #         loop.run_until_complete(loop.shutdown_asyncgens())
    #         loop.close()

    # @staticmethod # 11/8/24 switched to the bottom
    # def sync_wrapper(image_candidate, driver, index, n_queue):
    #     retries = 1
    #     for attempt in range(retries):
    #         try:
    #             # Create a task without needing asyncio.run()
    #             asyncio.run_coroutine_threadsafe(
    #                 image_candidate.download_image_with_selenium(driver, index, n_queue),
    #                 asyncio.get_event_loop()
    #             ).result()  # Ensure the coroutine completes in a thread-safe way
    #             if image_candidate.download_success and image_candidate.download_success not in ['skip', 'proxy']:
    #                 return  # Exit after the first successful attempt
    #         except Exception as e:
    #             print(f"Attempt {attempt + 1} failed with error: {e}")
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




    # async def handle_failures(self): #1
    #     while True:
    #         image_candidate = await self.failure_queue.get()
    #         if image_candidate is None:
    #             break  # Exit signal
    #         await self.retry_with_selenium(image_candidate)

    # async def retry_with_selenium(self, image_candidate): #NOOO
    #     for index, driver in enumerate(self.selenium_pool):
    #         success = await asyncio.get_event_loop().run_in_executor(self.executor, self.sync_wrapper, image_candidate, driver, index)
    #         if success:
    #             break
    # async def retry_with_selenium(self, image_candidate):#1
    #     index, driver = await self.driver_queue.get()
    #     # print(f"{bcolors.CWHITEBG}                RETRY {image_candidate.url} GBIF ID: {image_candidate.fullname}, Driver Index: {index}{bcolors.ENDC}")
    #     success = await asyncio.get_event_loop().run_in_executor(
    #         self.executor, self.sync_wrapper, image_candidate, driver, index
    #     )
    #     if success:
    #         print(f"{bcolors.OKCYAN}                SUCCESS{bcolors.ENDC}")
    #     else:
    #         await self.failure_queue.put(image_candidate)
    #     await self.driver_queue.put((index, driver))

    # @staticmethod
    # def sync_wrapper(image_candidate, driver, index):#1
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     result = loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index))
    #     loop.close()
    #     return result







    # async def fetch_image(self, session, image_row, semaphore):
    #     id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in image_row), None)
    #     if id_column is None:
    #         raise KeyError("No valid ID column found in image row.")

    #     print(f"Working on image: {image_row[id_column]}")
    #     gbif_id = image_row[id_column]
    #     gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['identifier']

    #     occ_row = self.find_gbifID(gbif_id)

    #     if occ_row is not None:
    #         image_candidate = ImageCandidate(self.cfg, image_row, occ_row, gbif_url, asyncio.Lock(), self.failure_log)
    #         async with semaphore:
    #             await image_candidate.download_image(session)
    #         return pd.DataFrame(occ_row) if image_candidate.download_success else None
    #     else:
    #         return None

    

    # async def process_batch(self):
    #     ssl_context = ssl.create_default_context(cafile=certifi.where())
    #     connector = TCPConnector(ssl=ssl_context)
    #     semaphore = asyncio.Semaphore(self.num_workers)  # Limit to 64 concurrent downloads max

    #     async with ClientSession(connector=connector) as session:
    #         tasks = []

    #         for _, image_row in self.images_df.iterrows():
    #             task = self.fetch_image(session, image_row, semaphore)
    #             tasks.append(task)

    #         results = await asyncio.gather(*tasks)

    #         for result in results:
    #             if result is not None:
    #                 self.results.append(result)

    #     if self.results:
    #         combined_results = pd.concat(self.results, ignore_index=True)
    #         self.save_results(combined_results)

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

        try:
            use_large = cfg['use_large_file_size_methods']
        except:
            use_large = False

        if use_large:
            occ_df = ImageBatchProcessor.ingest_DWC_large_files(filename_occ, dir_home)
            images_df = ImageBatchProcessor.ingest_DWC_large_files(filename_img, dir_home)
        else:
            occ_df = ImageBatchProcessor.ingest_DWC(filename_occ, dir_home)
            images_df = ImageBatchProcessor.ingest_DWC(filename_img, dir_home)
            
        
        return occ_df, images_df

    @staticmethod
    def ingest_DWC(DWC_csv_or_txt_file, dir_home):
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
    def ingest_DWC_large_files(file_path, dir_home, block_size=512):
        # Method to ingest large DWC (Darwin Core) CSV or TXT files using Dask to handle larger-than-memory files.
        # This function supports .txt or .csv files with varying delimiters.
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
            'organismRemarks': 'object'
        }
        # Check if the file name contains "occurrences" or "multimedia"
        if "occurrences" in file_name.lower() or "occurrence" in file_name.lower() or "multimedia" in file_name.lower():
            # Read the file without specifying dtypes to infer the column names
            # df_tmp = dd.read_csv(file_path, sep=None, header=0, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True)
            # df_tmp = dd.read_csv(file_path, sep=None, header=0, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True, nrows=5)
            # # Set all columns to dtype object (equivalent to strings in pandas/dask)
            # column_names = df_tmp.columns.tolist()
            # # column_dtypes = {col: 'object' for col in df_tmp.columns}
            # column_dtypes = {col: 'object' for col in column_names}

            # Read a small portion of the file to get the column names
            df_tmp = dd.read_csv(file_path, sep="\t", dtype='object',header=0, low_memory=False, blocksize=f"{block_size}MB", on_bad_lines="skip", assume_missing=True).head(n=5)
            column_names = df_tmp.columns.tolist()
            
            # Set all columns to dtype object (equivalent to strings in pandas/dask)
            column_dtypes = {col: 'object' for col in column_names}

        try:
            if "occurrences" in file_name.lower() or "occurrence" in file_name.lower() or "multimedia" in file_name.lower():
                if file_extension == 'txt':
                    # Reading a .txt file with tab-delimited data
                    df = dd.read_csv(file_path, sep="\t", header=0, dtype=column_dtypes, assume_missing=True, 
                                    blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                elif file_extension == 'csv':
                    # Reading a .csv file (comma-separated)
                    df = dd.read_csv(file_path, sep=",", header=0, dtype=column_dtypes, assume_missing=True, 
                                    blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                else:
                    # Handle other cases (e.g., pipe-separated or semicolon-separated)
                    try:
                        df = dd.read_csv(file_path, sep="|", header=0, dtype=column_dtypes, assume_missing=True, 
                                        blocksize=f"{block_size}MB", on_bad_lines="skip", low_memory=False)
                    except Exception:
                        try:
                            df = dd.read_csv(file_path, sep=";", header=0, dtype=column_dtypes, assume_missing=True, 
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

        return df.compute()





# class ImageBatchProcessor:
#     def __init__(self, cfg, images_df, occ_df=None):
#         self.cfg = cfg
#         self.images_df = images_df
#         self.occ_df = occ_df
#         self.num_workers = cfg.get('n_threads', 12)  # Default to 13 if not provided
#         self.queue = Queue()
#         self.results = []
#         self.lock = Lock()
#         self.threads = []

#     def start_worker_threads(self, worker_function):
#         for _ in range(self.num_workers):
#             thread = Thread(target=worker_function)
#             thread.start()
#             self.threads.append(thread)

#     def enqueue_tasks(self):
#         for _, image_row in self.images_df.iterrows():
#             self.queue.put(image_row)
#         self.queue.join()

#     def stop_workers(self):
#         for _ in range(self.num_workers):
#             self.queue.put(None)
#         for thread in self.threads:
#             thread.join()

#     def process_batch(self):
#         self.start_worker_threads(self.worker_download_standard)
#         self.enqueue_tasks()
#         self.stop_workers()
#         return self.results

#     def process_custom_batch(self):
#         self.start_worker_threads(self.worker_custom)
#         self.enqueue_tasks()
#         self.stop_workers()
#         return self.results

#     def worker_download_standard(self):
#         while True:
#             image_row = self.queue.get()
#             if image_row is None:
#                 self.queue.task_done()
#                 break  # None is the signal to stop processing

#             try:
#                 result = self.process_each_image_row(image_row)
#                 with self.lock:
#                     self.results.append(result)
#             except Exception as e:
#                 print(f"Error processing image: {e}")
#                 with self.lock:
#                     self.results.append(None)

#             self.queue.task_done()

#     def worker_custom(self):
#         while True:
#             image_row = self.queue.get()
#             if image_row is None:
#                 self.queue.task_done()
#                 break  # None is the signal to stop processing

#             try:
#                 result = self.process_each_custom_image_row(image_row)
#                 with self.lock:
#                     self.results.append(result)
#             except Exception as e:
#                 print(f"Error processing image: {e}")
#                 with self.lock:
#                     self.results.append(None)

#             self.queue.task_done()

#     def process_each_image_row(self, image_row):
#         id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in image_row), None)
#         if id_column is None:
#             raise KeyError("No valid ID column found in image row.")
        
#         print(f"Working on image: {image_row[id_column]}")
#         gbif_id = image_row[id_column]
#         gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['identifier']
        
#         occ_row = self.find_gbifID(gbif_id)

#         if occ_row is not None:
#             ImageInfo = ImageCandidate(self.cfg, image_row, occ_row, gbif_url, self.lock)
#             return pd.DataFrame(occ_row)
#         else:
#             return None

#     def process_each_custom_image_row(self, image_row):
#         col_url = self.cfg.get('col_url', 'identifier')
#         gbif_url = image_row[col_url]

#         print(f"Working on image: {image_row[self.cfg['col_name']]}")
#         if image_row is not None:
#             ImageInfo = ImageCandidateCustom(self.cfg, image_row, gbif_url, self.cfg['col_name'], self.lock)
#             return ImageInfo
#         else:
#             pass

#     def find_gbifID(self, gbif_id):
#         file_to_search = self.occ_df
#         if 'gbifID' in file_to_search.columns:
#             row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)), :]
#         elif 'id' in file_to_search.columns:
#             row_found = file_to_search.loc[file_to_search['id'].astype(str).str.match(str(gbif_id)), :]
#         elif 'coreid' in file_to_search.columns:
#             row_found = file_to_search.loc[file_to_search['coreid'].astype(str).str.match(str(gbif_id)), :]
#         else:
#             raise KeyError("Neither 'gbifID' nor 'id' found in the column names for the Occurrences file")

#         if row_found.empty:
#             print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
#             return None
#         else:
#             print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
#             return row_found


#     @staticmethod
#     def read_custom_file(cfg):
#         # Placeholder for reading the custom file logic.
#         pass

#     @staticmethod
#     def download_from_custom_file(cfg):
#         images_df = ImageBatchProcessor.read_custom_file(cfg)

#         col_url = cfg.get('col_url', 'identifier')
#         col_name = cfg['col_name']

#         # Report summary
#         print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
#         print(images_df.head())

#         # Ignore problematic Herbaria
#         if cfg['ignore_banned_herb']:
#             for banned_url in cfg['banned_url_stems']:
#                 images_df = images_df[~images_df[col_url].str.contains(banned_url, na=False)]

#         # Report summary
#         n_imgs = images_df.shape[0]
#         print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")

#         processor = ImageBatchProcessor(cfg, images_df)
#         return processor.process_custom_batch()
    
#     @staticmethod
#     def read_DWC_file(cfg):
#         dir_home = cfg['dir_home']
#         filename_occ = cfg['filename_occ']
#         filename_img = cfg['filename_img']
#         occ_df = ImageBatchProcessor.ingest_DWC(filename_occ, dir_home)
#         images_df = ImageBatchProcessor.ingest_DWC(filename_img, dir_home)
#         return occ_df, images_df

#     @staticmethod
#     def ingest_DWC(DWC_csv_or_txt_file, dir_home):
#         file_path = os.path.join(dir_home, DWC_csv_or_txt_file)
#         file_extension = DWC_csv_or_txt_file.split('.')[-1]

#         try:
#             if file_extension == 'txt':
#                 df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
#             elif file_extension == 'csv':
#                 try:
#                     df = pd.read_csv(file_path, sep=",", header=0, low_memory=False, dtype=str)
#                 except pd.errors.ParserError:
#                     try:
#                         df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
#                     except:
#                         try:
#                             df = pd.read_csv(file_path, sep="|", header=0, low_memory=False, dtype=str)
#                         except:
#                             df = pd.read_csv(file_path, sep=";", header=0, low_memory=False, dtype=str)
#             else:
#                 print(f"{bcolors.FAIL}DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened{bcolors.ENDC}")
#                 return None
#         except Exception as e:
#             print(f"Error while reading file: {e}")
#             return None

#         return df

    # @staticmethod
    # def check_n_images_in_group(detailedOcc, N):
    #     fam = detailedOcc['fullname'].unique()
    #     for f in fam:
    #         ct = len(detailedOcc[detailedOcc['fullname'].str.match(f)])
    #         if ct == N:
    #             print(f"{bcolors.OKGREEN}{f}: {ct}{bcolors.ENDC}")
    #         else:
    #             print(f"{bcolors.FAIL}{f}: {ct}{bcolors.ENDC}")



