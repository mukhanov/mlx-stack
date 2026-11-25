"""Standalone resumable download worker; exits if the API parent dies."""
import os
import sys
import threading
import time
from pathlib import Path

from . import hf


def main():
    parent = os.getppid()

    def watch_parent():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    hf.download(sys.argv[1], hf_hub=Path(sys.argv[2]))


if __name__ == "__main__":
    main()
