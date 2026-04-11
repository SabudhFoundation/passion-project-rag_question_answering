"""
src/preprocessing_data/pre_processing.py
==========================================
Orchestrator for the preprocessing stage.
Imports DataDownloader + Preprocessor and runs them in order.
No class definitions here — just the pipeline glue.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing_data.data_downloader import DataDownloader
from preprocessing_data.preprocessor    import Preprocessor
from utils import get_logger

logger = get_logger(__name__)


def run_preprocessing() -> list:
    """
    Runs DataDownloader → Preprocessor.
    Returns list of clean record dicts for the feature engineering stage.
    """
    downloader   = DataDownloader()
    file_path    = downloader.download()
    if not file_path:
        raise RuntimeError(
            "DataDownloader returned no file path.\n"
            "Cannot continue to preprocessing. Check download errors above."
        )

    preprocessor = Preprocessor()
    records      = preprocessor.process(file_path)
    if not records:
        raise RuntimeError(
            "Preprocessor returned 0 records.\n"
            "Check the HotpotQA file is valid and not empty."
        )

    logger.info(f"Preprocessing stage complete: {len(records):,} records")
    return records


if __name__ == "__main__":
    try:
        records = run_preprocessing()
        print(f"\n✅ Preprocessing done. {len(records):,} records ready.")
    except RuntimeError as e:
        print(f"\n❌ {e}")
