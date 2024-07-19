import os, json
import sys
import inspect
import certifi
import ssl
import pandas as pd
import asyncio
from aiohttp import ClientSession, TCPConnector
import undetected_chromedriver as uc
from concurrent.futures import ThreadPoolExecutor

from leafmachine2.machine.general_utils import bcolors, validate_dir
from leafmachine2.downloading.utils_downloads_candidate import ImageCandidate, ImageCandidateCustom

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)


async def download_all_images_in_images_csv(cfg):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']

    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)

    if cfg['is_custom_file']:
        ImageBatchProcessor.download_from_custom_file(cfg)
        return

    occ_df, images_df = ImageBatchProcessor.read_DWC_file(cfg)

    if occ_df is None:
        return

    # Report summary
    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())
    print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
    print(occ_df.head())

    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]

    occ_id_column = next((col for col in ['id', 'gbifID'] if col in occ_df.columns), None)
    images_id_column = next((col for col in ['coreid', 'gbifID'] if col in images_df.columns), None)

    if not (occ_id_column and images_id_column):
        raise ValueError("Required ID columns not found in dataframes.")

    occ_df[occ_id_column] = occ_df[occ_id_column].astype(str)
    images_df[images_id_column] = images_df[images_id_column].astype(str)
    images_df = images_df[images_df[images_id_column].isin(occ_df[occ_id_column])]

    # Report summary
    print(f"{bcolors.BOLD}Number of images in images file: {images_df.shape[0]}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Number of occurrence to search through: {occ_df.shape[0]}{bcolors.ENDC}")

    processor = ImageBatchProcessor(cfg, images_df, occ_df)
    processor.init_selenium_drivers()  # Initialize 4 Selenium drivers

    # asyncio.run(processor.process_batch())
    # await asyncio.gather(processor.process_batch(),processor.handle_failures())
    await processor.process_batch()

    processor.close_selenium_drivers()


class ImageBatchProcessor:
    def __init__(self, cfg, images_df, occ_df=None):
        self.cfg = cfg
        self.images_df = images_df
        self.occ_df = occ_df
        self.results = []
        self.num_workers = self.cfg.get('n_threads', 64)
        self.num_drivers = self.cfg.get('num_drivers', 8)
        self.failure_log = {}

        self.failure_queue = asyncio.Queue()
        self.selenium_pool = []
        self.driver_queue = asyncio.Queue()

        self.n_queue = 0
        self.total_rows = len(images_df)
        self.processed_rows = 0
        self.lock = asyncio.Lock()

        self.executor = ThreadPoolExecutor(self.num_drivers)

    def init_selenium_drivers(self):
        for index in range(self.num_drivers):
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            driver = uc.Chrome(options=options)
            self.selenium_pool.append(driver)
            self.driver_queue.put_nowait((index, driver))

    def close_selenium_drivers(self):
        for driver in self.selenium_pool:
            driver.quit()
        self.selenium_pool = []

    async def fetch_image(self, session, image_row, semaphore):
        id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in image_row), None)
        if id_column is None:
            raise KeyError("No valid ID column found in image row.")

        print(f"Working on image: {image_row[id_column]}")
        gbif_id = image_row[id_column]
        gbif_url = image_row[self.cfg['custom_url_column_name']] if self.cfg['is_custom_file'] else image_row['identifier']

        occ_row = self.find_gbifID(gbif_id)

        if occ_row is not None:
            image_candidate = ImageCandidate(self.cfg, image_row, occ_row, gbif_url, asyncio.Lock(), self.failure_log)
            async with semaphore:
                await image_candidate.download_image(session, self.n_queue)
            if image_candidate.download_success == '404':
                print(f"Skipping 404 error for image: {image_row[id_column]}")
                return None
            if not image_candidate.download_success:
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
        
    async def process_batch(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(ssl=ssl_context)
        semaphore = asyncio.Semaphore(self.num_workers - self.num_drivers)  # Reserve num_drivers threads for Selenium

        async with ClientSession(connector=connector) as session:
            tasks = []

            for _, image_row in self.images_df.iterrows():
                task = self.fetch_image(session, image_row, semaphore)
                tasks.append(task)
            
            await asyncio.gather(*tasks)

        await self.handle_failures() 

        if self.results:
            combined_results = pd.concat(self.results, ignore_index=True)
            self.save_results(combined_results)

        await self.finalize()

        self.save_failure_log()

    async def handle_failures(self):
        driver_tasks = [self.process_with_driver(index, driver) for index, driver in enumerate(self.selenium_pool)]
        await asyncio.gather(*driver_tasks)
        print(f"handle_failures completed. Queue size: {self.failure_queue.qsize()}")


    async def process_with_driver(self, index, driver):
        while True:
            try:
                image_candidate = await self.failure_queue.get()
                if image_candidate is None:
                    print(f"Driver {index} received exit signal")
                    self.failure_queue.task_done()
                    break  # Exit signal
                # print(f"{bcolors.CWHITEBG}                RETRY {image_candidate.url} GBIF ID: {image_candidate.fullname}, Driver Index: {index}{bcolors.ENDC}")
                self.n_queue = self.failure_queue.qsize()
                if self.n_queue > 0:
                    success = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.sync_wrapper, image_candidate, driver, index, self.n_queue
                    )
                    if not success:
                        await self.failure_queue.put(image_candidate)
                    self.failure_queue.task_done()
                    print(f"Queue size after processing by driver {index}: {self.failure_queue.qsize()}")

                async with self.lock:
                    if self.processed_rows >= self.total_rows:
                        if self.failure_queue.empty():
                            for _ in range(self.num_drivers):
                                await self.failure_queue.put(None)
            except asyncio.CancelledError:
                break

    @staticmethod
    def sync_wrapper(image_candidate, driver, index, n_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(image_candidate.download_image_with_selenium(driver, index, n_queue))
        loop.close()
        return result
    
    async def finalize(self):
        # Ensure all failures are processed
        # while not self.failure_queue.empty():
        #     await asyncio.sleep(1)  # Small delay to ensure all tasks are processed
        #     print(f"Failure queue size: {self.failure_queue.qsize()}")

        self.close_selenium_drivers()
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
                df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
            elif file_extension == 'csv':
                try:
                    df = pd.read_csv(file_path, sep=",", header=0, low_memory=False, dtype=str)
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
                    except:
                        try:
                            df = pd.read_csv(file_path, sep="|", header=0, low_memory=False, dtype=str)
                        except:
                            df = pd.read_csv(file_path, sep=";", header=0, low_memory=False, dtype=str)
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



