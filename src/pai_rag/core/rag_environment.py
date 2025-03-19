import os
from pai_rag.app.web.web_lock import DEFAULT_LOCK_FILE_PATH, WebInstanceLock


class RagServiceEnvironment:
    def __init__(self):
        self.IS_API_INSTANCE = os.getenv("DEPLOY_MODE", "web").upper() == "API"
        self.SHOULD_START_WEB = not self.IS_API_INSTANCE
        self.lock_file = open(DEFAULT_LOCK_FILE_PATH, "w")

    def acquire_lock(self):
        if self.SHOULD_START_WEB:
            lock_success = WebInstanceLock.try_acquire(self.lock_file, timeout=2)
            self.SHOULD_START_WEB = lock_success

    def cleanup(self):
        if self.SHOULD_START_WEB:
            WebInstanceLock.release(self.lock_file)
        self.lock_file.close()


service_environment = RagServiceEnvironment()
