import fcntl
import os
import time
from loguru import logger

INSTANCE_SERVICE_NAME = os.environ.get("SERVICE_NAME", "default")
DEFAULT_LOCK_FILE_PATH = "localdata/__web_instance_lock__{INSTANCE_SERVICE_NAME}__.lock"


class WebInstanceLock:
    @staticmethod
    def try_acquire(file_handler, timeout=2):
        start_time = time.time()
        while True:
            try:
                # 尝试获取排它锁，使用 LOCK_NB 进行非阻塞操作
                fcntl.flock(file_handler, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info("Acquire lock successfully.")
                return True
            except BlockingIOError:
                # 被锁住，检查是否超时
                if time.time() - start_time > timeout:
                    logger.info("Acquire lock failed.")
                    return False
                time.sleep(0.1)  # 短暂等待再尝试

    @staticmethod
    def release(file_handler):
        fcntl.flock(file_handler, fcntl.LOCK_UN)  # 解
        logger.info("Release lock successfully.")
