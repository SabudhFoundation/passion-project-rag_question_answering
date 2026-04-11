"""
src/preprocessing_data/data_downloader.py
==========================================
CLASS: DataDownloader

ONE CLASS — ONE FILE.
SINGLE RESPONSIBILITY: Only downloads HotpotQA. Nothing else.

EXCEPTION HANDLING IN THIS FILE:
  Every operation that can fail has a try-except block.
  Each exception gives a clear message telling you exactly what went wrong
  and how to fix it — not just "Error occurred".
"""

import os
import sys
import urllib.request
import urllib.error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, ensure_dir, file_exists_and_valid, print_section

# Module-level logger — uses the filename as the logger name
logger = get_logger(__name__)


class DataDownloader:
    """
    Downloads the HotpotQA dataset to src/data/raw/.

    USAGE:
        downloader = DataDownloader()
        path       = downloader.download()

    RETURNS:
        str  — local file path on success
        None — if download failed (error already logged)
    """

    def __init__(self):
        try:
            ensure_dir(config.RAW_DIR)
            logger.info(f"DataDownloader ready. Output: {config.RAW_DIR}")
        except OSError as e:
            # If we can't even create the folder, raise immediately
            raise OSError(
                f"Cannot create data directory: {config.RAW_DIR}\n"
                f"Reason: {e}\n"
                f"Check that you have write permission to this location."
            )

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def download(self) -> str:
        """
        Downloads HotpotQA training set from CMU servers.

        Returns:
            str  — path to downloaded file
            None — if download failed
        """
        print_section("STEP 1/4 — DataDownloader")

        # Check if already downloaded (skip if yes)
        if self._already_downloaded():
            return config.HOTPOTQA_FILE

        logger.info("Starting HotpotQA download (~540 MB)...")
        logger.info(f"URL: {config.HOTPOTQA_URL}")

        try:
            urllib.request.urlretrieve(
                config.HOTPOTQA_URL,
                config.HOTPOTQA_FILE,
                self._progress_hook,
            )
            mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
            print()  # new line after progress bar
            logger.info(f"Download complete ({mb:.0f} MB) → {config.HOTPOTQA_FILE}")
            return config.HOTPOTQA_FILE

        except urllib.error.URLError as e:
            # Network/DNS errors — no internet, URL wrong, server down
            logger.error(
                f"Network error during download: {e}\n"
                f"  → Check your internet connection.\n"
                f"  → Try opening {config.HOTPOTQA_URL} in your browser.\n"
                f"  → If it opens, save the file manually to: {config.HOTPOTQA_FILE}"
            )
            return None

        except urllib.error.HTTPError as e:
            # HTTP errors — 404, 403, 500 etc.
            logger.error(
                f"HTTP error {e.code} when downloading HotpotQA: {e.reason}\n"
                f"  → The server returned an error. Try again later.\n"
                f"  → URL: {config.HOTPOTQA_URL}"
            )
            return None

        except OSError as e:
            # Disk write errors — no space, permission denied
            logger.error(
                f"Could not save file to disk: {e}\n"
                f"  → Check disk space (need ~600 MB free).\n"
                f"  → Check write permission for: {config.RAW_DIR}"
            )
            return None

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _already_downloaded(self) -> bool:
        """
        Checks if file exists and is large enough to be valid.
        Prevents re-downloading 540 MB every pipeline run.
        """
        if file_exists_and_valid(config.HOTPOTQA_FILE, min_mb=100):
            mb = os.path.getsize(config.HOTPOTQA_FILE) / (1024 * 1024)
            logger.info(f"HotpotQA already downloaded ({mb:.0f} MB) — skipping")
            return True
        return False

    def _progress_hook(self, block_num: int, block_size: int,
                       total_size: int) -> None:
        """Shows download % in terminal while downloading."""
        if total_size > 0:
            downloaded = block_num * block_size
            pct        = min(downloaded / total_size * 100, 100)
            mb         = downloaded / (1024 * 1024)
            print(f"\r  Progress: {pct:.1f}% ({mb:.0f} MB)",
                  end="", flush=True)


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    downloader = DataDownloader()
    path       = downloader.download()
    if path:
        print(f"\n✅ Ready at: {path}")
    else:
        print("\n❌ Download failed. See error messages above.")
