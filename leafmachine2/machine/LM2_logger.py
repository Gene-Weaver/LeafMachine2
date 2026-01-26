import logging, os, psutil, torch, platform, cpuinfo, yaml,shutil #py-cpuinfo
from leafmachine2.machine.general_utils import get_datetime

def start_logging(Dirs, cfg):
    run_name = cfg['leafmachine']['project']['run_name']
    path_log = os.path.join(Dirs.path_log, '__'.join(['LM2-log',str(get_datetime()), run_name])+'.log')

    # Disable default StreamHandler
    logging.getLogger().handlers = []

    # create logger
    logger = logging.getLogger('Hardware Components')
    logger.setLevel(logging.DEBUG)

    # create file handler and set level to debug
    fh = logging.FileHandler(path_log)
    fh.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # add formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Create a logger for the file handler
    file_logger = logging.getLogger('file_logger')
    file_logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path_log)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    # Disable propagation of log messages to the root logger
    file_logger.propagate = False

    # 'application' code
    # logger.debug('debug message')
    # logger.info('info message')
    # logger.warning('warn message')
    # logger.error('error message')
    # logger.critical('critical message')

    # Get CPU information
    logger.info(f"CPU: {find_cpu_info()}")

    # Get GPU information (using PyTorch)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            gpu = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu.name} ({gpu.total_memory // (1024 * 1024)} MB)")
        else:
            for i in range(num_gpus):
                gpu = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {gpu.name} ({gpu.total_memory // (1024 * 1024)} MB)")
    else:
        logger.info("No GPU found")

    # Get memory information
    mem_info = psutil.virtual_memory()
    logger.info(f"Memory: {mem_info.total // (1024 * 1024)} MB")
    logger.info(LM2_banner())
    logger.info(f"Config added to log file")
    file_logger.info('Config:\n{}'.format(yaml.dump(cfg)))


    return logger


def start_worker_logging(worker_id, Dirs, log_name):
    worker_log_path = os.path.join(Dirs.path_log, f'{log_name}_{worker_id}.log')

    logger_name = f'Worker_{worker_id}'
    worker_logger = logging.getLogger(logger_name)
    worker_logger.setLevel(logging.DEBUG)

    # Critical: prevent double-printing via root logger if you also configure root
    worker_logger.propagate = False

    # If we've already configured this logger, reuse it
    if worker_logger.handlers:
        return worker_logger, worker_log_path


    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # File handler (only add if not already present for this file)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(worker_log_path)
        for h in worker_logger.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(worker_log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        worker_logger.addHandler(fh)

    # Console handler (optional) — add only once
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                             for h in worker_logger.handlers)
    if not has_stream_handler:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        worker_logger.addHandler(ch)

    worker_logger._lm2_configured = True
    return worker_logger, worker_log_path

def merge_worker_logs(Dirs, num_workers, main_log_name, log_name_stem):
    # Merge worker logs into the main log file
    main_log_path = os.path.join(Dirs.path_log, f'{main_log_name}.log')

    with open(main_log_path, 'a') as main_log_file:
        for worker_id in range(num_workers):
            worker_log_path = os.path.join(Dirs.path_log, f'{log_name_stem}_{worker_id}.log')
            if os.path.exists(worker_log_path):
                with open(worker_log_path, 'r') as worker_log_file:
                    shutil.copyfileobj(worker_log_file, main_log_file)  # Append worker log to the main log
                os.remove(worker_log_path)  # Optionally delete the worker log file after merging


def find_cpu_info():
    cpu_info = []
    cpu_info.append(platform.processor())
    try:

        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_info.append(line.split(':')[1].strip())
                    break
        return ' / '.join(cpu_info)
    except:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_info = []
            cpu_info.append(info['brand_raw'])
            cpu_info.append(f"{info['hz_actual_friendly']}")
            return ' / '.join(cpu_info)
        except:
            return "CPU: UNKNOWN"

def LM2_banner():
        logo = """
       __             __                   _     _            ____  
      / /  ___  __ _ / _| /\/\   __ _  ___| |__ (_)_ __   ___|___ \ 
     / /  / _ \/ _` | |_ /    \ / _` |/ __| '_ \| | '_ \ / _ \ __) |
    / /__|  __/ (_| |  _/ /\/\ \ (_| | (__| | | | | | | |  __// __/ 
    \____/\___|\__,_|_| \/    \/\__,_|\___|_| |_|_|_| |_|\___|_____| """
        return logo