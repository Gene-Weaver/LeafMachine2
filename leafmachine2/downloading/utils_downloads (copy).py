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

from leafmachine2.machine.general_utils import bcolors, validate_dir
from leafmachine2.downloading.utils_downloads_candidate import ImageCandidate#, PrettyPrint

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)


# def run_download_parallel(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker):
#     # pp = PrettyPrint()
#     pp = None
#     asyncio.run(run_main(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp))
def run_download_parallel(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker):
    # pp = PrettyPrint()
    pp = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_main(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

async def run_main(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp):
    await download_all_images_in_images_csv_selenium(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp)

async def download_all_images_in_images_csv_selenium(cfg, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp):
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
    
    # Shuffle the DataFrame with a random seed
    occ_df = occ_df.sample(frac=1, random_state=2023).reset_index(drop=True)

    # Sort by the frequency of fullname_index
    fullname_counts = occ_df['fullname_index'].value_counts().to_dict()
    occ_df['fullname_frequency'] = occ_df['fullname_index'].map(fullname_counts)
    occ_df = occ_df.sort_values(by='fullname_frequency', ascending=True).reset_index(drop=True)
    occ_df = occ_df.drop(columns=['fullname_frequency'])


    print(f"{bcolors.BOLD}Number of images in images file: {images_df.shape[0]}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Number of occurrence to search through: {occ_df.shape[0]}{bcolors.ENDC}")

    processor = ImageBatchProcessor(cfg, images_df, occ_df, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp)
    processor.init_selenium_drivers()

    try:
        await processor.process_batch()
    finally:
        await processor.finalize()

class ImageBatchProcessor:
    def __init__(self, cfg, images_df, occ_df, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, pp):
        self.cfg = cfg
        self.images_df = images_df
        self.occ_df = occ_df
        self.results = []
        self.num_workers = self.cfg.get('n_threads', 8)
        self.num_drivers = self.cfg.get('num_drivers', 8)
        self.failure_log = {}

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

        self.download_tracker = download_tracker
        self.completed_tracker = completed_tracker
        self.banned_url_tracker = banned_url_tracker
        self.banned_url_counts_tracker = banned_url_counts_tracker
        
        self.pp = pp
        self.shutdown_flag = asyncio.Event()  # Add the shutdown flag
        # self.download_tracker = {'names': download_tracker, 'completed': set()}

    def get_driver_with_random_user_agent(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:86.0) Gecko/20100101 Firefox/86.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
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

    def init_proxy_drivers(self):
        for index in range(5):  # Initialize 5 proxy drivers
            driver = self.get_driver_with_random_user_agent()
            self.proxy_pool.append(driver)
            self.proxy_queue.put_nowait((index, driver))

    def close_selenium_drivers(self):
        while not self.driver_queue.empty():
            _, driver = self.driver_queue.get_nowait()
            driver.quit()
        self.selenium_pool = []

    def close_proxy_drivers(self):
        while not self.proxy_queue.empty():
            _, driver = self.proxy_queue.get_nowait()
            driver.quit()
        self.proxy_pool = []

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

                image_candidate = ImageCandidate(self.cfg, occ_row, image_row, gbif_url, asyncio.Lock(), self.failure_log, self.download_tracker, self.completed_tracker, self.banned_url_tracker, self.banned_url_counts_tracker)#, self.pp)
                async with semaphore:
                    print(f"Working on image: {image_row[id_column]}")
                    # if isinstance(gbif_url, str) and gbif_url.startswith('data:image'): # THis is duplicated
                    #     await image_candidate.download_base64_image(gbif_url)
                    # elif isinstance(gbif_url, str):
                    await image_candidate.download_image(session, self.n_queue, logging_enabled=False)
                    self.remove_fullnames_with_all_images(image_candidate.fullname)
                    # else:
                        # pass

                    if image_candidate.download_success == 'skip':
                        print(f"Skipping 404 error for image: {image_row[id_column]}")
                    elif image_candidate.download_success == False:
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
        except Exception as e:
            print(f"Error in fetch_image_with_proxy: {e}")
    '''
    # This code is for starting with image, matching to occ. But it's more efficient now to to the inverse
    async def fetch_image(self, session, image_row, semaphore):
        try:
            id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in image_row), None)
            if id_column is None:
                raise KeyError("No valid ID column found in image row.")
            
            gbif_id = image_row[id_column]
            gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['identifier']

            occ_row = self.find_gbifID(gbif_id)

            if occ_row is not None:
                image_candidate = ImageCandidate(self.cfg, image_row, occ_row, gbif_url, asyncio.Lock(), self.failure_log, self.download_tracker)
                async with semaphore:
                    print(f"Working on image: {image_row[id_column]}")
                    if isinstance(gbif_url, str) and gbif_url.startswith('data:image'):
                        await image_candidate.download_base64_image(gbif_url)
                    elif isinstance(gbif_url, str):
                        await image_candidate.download_image(session, self.n_queue, logging_enabled=False)
                        # Check and remove fullnames that have all images downloaded
                        self.remove_fullnames_with_all_images(image_candidate.fullname)
                    else:
                        pass
                        # raise ValueError(f"                If nan, then occ has no image --> {gbif_url}")
                    
                if image_candidate.download_success == '404':
                    print(f"Skipping 404 error for image: {image_row[id_column]}")
                elif not image_candidate.download_success:
                    await self.failure_queue.put(image_candidate)
                    self.n_queue = self.failure_queue.qsize()
                    print(f"{bcolors.CBLACKBG}QUEUE [{self.n_queue}]{bcolors.ENDC}")

                async with self.lock:
                    self.processed_rows += 1

                return pd.DataFrame(occ_row) if image_candidate.download_success else None
            else:
                async with self.lock:
                    self.processed_rows += 1
                return None
        except Exception as e:
            print(f"Error in fetch_image: {e}")
            return None
    '''

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

    # def remove_fullnames_with_all_images(self):
    #     fullnames_to_remove = []
    #     for fullname, count in self.download_tracker.items():
    #         if count >= self.cfg['n_to_download']:
    #             fullnames_to_remove.append(fullname)
        
    #     for fullname in fullnames_to_remove:
    #         self.occ_df = self.occ_df[self.occ_df['fullname_index'] != fullname]
    #         print(f"{bcolors.CGREYBG}REMOVED [{fullname}] from occ_df{bcolors.ENDC}")


    # def remove_fullnames_with_all_images(self, fullname):
    #     self.occ_df = self.occ_df[self.occ_df['fullname_index'] != fullname]
    #     print(f"{bcolors.CGREYBG}REMOVED [{fullname}] from occ_df{bcolors.ENDC}")

    async def producer(self, queue):
        print(f"Producer: Starting to iterate over occ_df")
        for _, occ_row in self.occ_df.iterrows():
            await queue.put(occ_row)
        for _ in range(self.num_workers):
            await queue.put(None)  # Stop signal for consumers
        print(f"Producer: Finished adding items to the queue")
    '''
    # This code is for starting with image, matching to occ. But it's more efficient now to to the inverse
    async def producer(self, queue):
        print(f"Producer: Starting to iterate over images_df")
        for _, image_row in self.images_df.iterrows():
            await queue.put(image_row)
        for _ in range(self.num_workers):
            await queue.put(None)  # Stop signal for consumers
        print(f"Producer: Finished adding items to the queue")
    '''

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
            except Exception as e:
                print(f"Failure Consumer {index} encountered an error: {e}")
                continue  # Ensure the consumer keeps running even if an error occurs

        print(f"Failure Consumer {index}: Finished processing items")

    async def proxy_consumer(self, session, proxy_queue, semaphore_proxy):
        print(f"Proxy Consumer: Waiting for items")
        while True:
            image_candidate = await proxy_queue.get()
            if image_candidate is None:
                proxy_queue.task_done()  # Ensure task_done is called for the stop signal
                print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
                print(f"{bcolors.CGREENBG}Proxy Consumer: received shutdown signal.{bcolors.ENDC}")
                print(f"{bcolors.CGREENBG}***************************************************{bcolors.ENDC}")
                break

            async with semaphore_proxy:
                await self.fetch_image_with_proxy(session, image_candidate)

            proxy_queue.task_done()
        print(f"Proxy Consumer: Finished processing items")


    #*** global flag
    # async def failure_consumer(self, index, driver):
    #     while True:
    #         image_candidate = await self.failure_queue.get()
    #         if image_candidate is None:
    #             self.failure_queue.task_done()
    #             print(f"Failure Consumer {index}: received shutdown signal.")
    #             break
    #         await self.process_with_driver(image_candidate, driver, index)
    #         self.failure_queue.task_done()
    #     print(f"Failure Consumer {index}: Finished processing items")

    async def process_batch(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(ssl=ssl_context)

        n_semaphore = self.num_workers
        if n_semaphore <= 2:
            n_semaphore = self.num_workers
        semaphore = asyncio.Semaphore(n_semaphore)

        semaphore_proxy = asyncio.Semaphore(5)  # Limit the number of concurrent proxy consumers
        queue = asyncio.Queue(maxsize=1000)
        proxy_queue = asyncio.Queue(maxsize=1000)  # Queue for proxy consumers

        async with ClientSession(connector=connector) as session:
            producers = [asyncio.create_task(self.producer(queue))]
            consumers = [asyncio.create_task(self.consumer(session, queue, semaphore)) for _ in range(self.num_workers)]
            failure_consumers = [asyncio.create_task(self.failure_consumer(index, driver)) for index, driver in enumerate(self.selenium_pool)]
            proxy_consumers = [asyncio.create_task(self.proxy_consumer(session, proxy_queue, semaphore_proxy)) for _ in range(5)]

            await asyncio.gather(*producers)
            await queue.join()

            for _ in range(self.num_workers):
                await queue.put(None)  # Stop signal for consumers
            await asyncio.gather(*consumers)
            print(f"{bcolors.CGREENBG}All consumers have finished processing.{bcolors.ENDC}")

            self.shutdown_flag.set()

            # Add shutdown signals for failure consumers
            for _ in range(self.num_drivers):
                await self.failure_queue.put(None)

            await self.failure_queue.join()  # Ensure all failure tasks are processed

            await asyncio.gather(*failure_consumers)
            print(f"{bcolors.CGREENBG}All failure consumers have shut down.{bcolors.ENDC}")

            # Add shutdown signals for proxy consumers
            for _ in range(5):  # 5 proxy consumers
                await proxy_queue.put(None)

            await proxy_queue.join()  # Ensure all proxy tasks are processed

            await asyncio.gather(*proxy_consumers)
            print(f"{bcolors.CGREENBG}All proxy consumers have shut down.{bcolors.ENDC}")

            await self.finalize()



    # await self.monitor_failure_queue(failure_consumers)
    #*** global flag
    # async def process_batch(self):
    #     # self.pp.start()
    #     ssl_context = ssl.create_default_context(cafile=certifi.where())
    #     connector = TCPConnector(ssl=ssl_context)

    #     n_semaphore = self.num_workers
    #     if n_semaphore <= 2:
    #         n_semaphore = self.num_workers
    #     semaphore = asyncio.Semaphore(n_semaphore)

    #     queue = asyncio.Queue(maxsize=100)

    #     async with ClientSession(connector=connector) as session:
    #         producers = [asyncio.create_task(self.producer(queue))]
    #         consumers = [asyncio.create_task(self.consumer(session, queue, semaphore)) for _ in range(self.num_workers)]
    #         failure_consumers = [asyncio.create_task(self.failure_consumer(index, driver)) for index, driver in enumerate(self.selenium_pool)]

    #         await asyncio.gather(*producers)
    #         await queue.join()
    #         # await asyncio.gather(*consumers) ##########
    #         for _ in range(self.num_workers): ############
    #             await queue.put(None)  # Stop signal for consumers ############
    #         await asyncio.gather(*consumers) ############
    #         print(f"{bcolors.CGREENBG}All consumers have finished processing.{bcolors.ENDC}")


    #         for _ in range(self.num_drivers):
    #             await self.failure_queue.put(None)  # Stop signal for failure consumers
    #         await self.failure_queue.join()  # Ensure all failure tasks are processed 
    #         print(f"{bcolors.CGREENBG}All failure queue items have been processed.{bcolors.ENDC}")

    #         await asyncio.gather(*failure_consumers)
    #         print(f"{bcolors.CGREENBG}All failure consumers have shut down.{bcolors.ENDC}")
    #         # self.pp.stop()

    #     await self.finalize()

    #*** global flag
    # async def monitor_failure_queue(self, failure_consumers):
    #     while True:
    #         if self.failure_queue.empty():
    #             await asyncio.sleep(1)  # Small delay to prevent busy waiting
    #         else:
    #             if all(consumer.done() for consumer in failure_consumers):
    #                 failure_consumers = [asyncio.create_task(self.failure_consumer(index, driver)) for index, driver in enumerate(self.selenium_pool)]

    #         if all(consumer.done() for consumer in failure_consumers) and self.failure_queue.empty():
    #             break

    #     print(f"{bcolors.CGREENBG}All failure queue items have been processed and all consumers have shut down.{bcolors.ENDC}")

    #     # for _ in range(self.num_drivers):
    #     #     await self.failure_queue.put(None)  # Stop signal for failure consumers
    #     # self.shutdown_flag.set()  # Set the shutdown flag
    #     # await self.failure_queue.join()  # Ensure all failure tasks are processed
    #     self.shutdown_flag.set()  # Set the shutdown flag
    #     await self.failure_queue.join()  # Ensure all failure tasks are processed

    #     await asyncio.gather(*failure_consumers)
    #     print(f"{bcolors.CGREENBG}All failure consumers have shut down.{bcolors.ENDC}")


    async def process_with_driver(self, image_candidate, driver, index):
        active_drivers_count = sum(1 for d in self.selenium_pool if d.service.process is not None and d.service.process.poll() is None)
        
        print(f"{bcolors.CWHITEBG}[Active drivers {active_drivers_count}]{bcolors.ENDC}")
        
        success = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.sync_wrapper, image_candidate, driver, index, self.n_queue
        )
        n_queue = self.failure_queue.qsize()        
        n_queue_proxy = self.proxy_queue.qsize()

        if not success and image_candidate.download_success == 'proxy':
            await self.proxy_queue.put(image_candidate)
            print(f"Added to proxy queue for image: {image_candidate.fullname}. PROXY QUEUE REMAINING {n_queue_proxy}")
        elif not success:
            print(f"Failed processing with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")
        else:
            print(f"Successfully processed with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")


    

    #*** global flag
    # async def process_with_driver(self, image_candidate, driver, index):
    #     success = await asyncio.get_event_loop().run_in_executor(
    #         self.executor, self.sync_wrapper, image_candidate, driver, index, self.n_queue
    #     )
    #     n_queue = self.failure_queue.qsize()
    #     if not success:
    #         print(f"Failed processing with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")
    #     else:
    #         print(f"Successfully processed with driver {index} for image: {image_candidate.fullname}. QUEUE REMAINING {n_queue}")

    @staticmethod
    def sync_wrapper(image_candidate, driver, index, n_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        retries = 1
        backoff = 1
        for attempt in range(retries):
            result = loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
            if result:
                break
            time.sleep(backoff * (2 ** attempt))
        loop.close()
        return result

    @staticmethod
    def proxy_sync_wrapper(image_candidate, driver, index, n_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        retries = 3
        backoff = 1
        for attempt in range(retries):
            result = loop.run_until_complete(image_candidate.download_image_with_proxy(driver, index, n_queue))
            if result:
                break
            time.sleep(backoff * (2 ** attempt))
        loop.close()
        return result


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



