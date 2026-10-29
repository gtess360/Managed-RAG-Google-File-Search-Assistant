r"""Corpus ingestion script for the Personal AI Knowledge Assistant.

Uploads all supported documents from data/documents/ into a Google Gemini
File Search store. Google handles chunking, embedding and indexing server-side.

Usage:
    .\.venv\Scripts\python.exe ingest.py                 # Mirror local docs into the store
    .\.venv\Scripts\python.exe ingest.py --no-replace    # Append without clearing the store
    .\.venv\Scripts\python.exe ingest.py --allow-partial-ingestion

Requires GEMINI_API_KEY in the environment or root .env file.

POLICY
------
By default every run REPLACES the store contents so the store mirrors the local corpus.
Do NOT run this script as part of the test suite.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running from the project root without src/ on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PATHS
from src.file_search import list_local_documents, sync_documents


def ingest_corpus(
    create_store: bool = False,
    replace: bool = False,
    dry_run: bool = False,
    allow_partial_ingestion: bool = False,
    data_dir: Optional[Path] = None,
    storage_dir: Optional[Path] = None,  # deprecated, ignored
    force_rebuild: bool = False,  # deprecated, ignored
    allow_empty: bool = False,  # deprecated, ignored
) -> dict:
    """Upload the local corpus to the Gemini File Search store safely.

    Guarantees:
    1. Synchronization is NON-DESTRUCTIVE by default (replace=False).
    2. Does NOT delete existing store documents before uploading; existing indexed documents
       remain intact if an upload or indexing operation fails.
    3. Handles duplicate and changed documents explicitly; reports planned changes.
    4. Never treats partial ingestion as complete unless allow_partial_ingestion=True is set.

    Args:
        create_store: Explicitly create a new store for this project if not already configured.
        replace: Replace changed/duplicate documents (non-destructive by default).
        dry_run: Inspect and report what would happen without modifying the store.
        allow_partial_ingestion: Proceed even if some files fail to upload.
        data_dir: Directory to read documents from (defaults to PATHS.data_dir).
        storage_dir: Deprecated; ignored.
        force_rebuild: Deprecated; ignored.
        allow_empty: Deprecated; an empty corpus is always refused.

    Returns:
        Summary dict (store_name, display_name, uploaded, skipped, removed, failed, completed).

    Raises:
        ValueError: If no supported documents are found, or no store is configured and create_store=False.
        RuntimeError: If uploads fail without allow_partial_ingestion.
    """
    target_data_dir = data_dir or PATHS.data_dir

    print("=" * 60)
    print("Personal AI Knowledge Assistant — Gemini File Search Ingestion")
    print("=" * 60)

    files = list_local_documents(target_data_dir)
    print(f"\n[1/2] Found {len(files)} supported document(s) in: {target_data_dir}")
    if not files:
        raise ValueError("No documents were found and corpus is empty. Aborting.")

    mode_desc = "dry-run" if dry_run else ("replace" if replace else "non-destructive sync")
    print(f"\n[2/2] Synchronizing with File Search store (mode={mode_desc}, create_store={create_store}) ...")

    summary = sync_documents(
        data_dir=target_data_dir,
        replace=replace,
        create_store_if_missing=create_store,
        allow_partial=allow_partial_ingestion,
        dry_run=dry_run,
    )

    if summary.get("dry_run"):
        print("\nDry-run complete. No changes were made to the remote store.")
        return summary

    if summary["failed"]:
        print(f"\n  Upload/indexing failures ({len(summary['failed'])}):")
        for item in summary["failed"]:
            print(f"    ! {item}")
        if not allow_partial_ingestion:
            raise RuntimeError(
                f"{len(summary['failed'])} file(s) failed to upload or index. "
                "Previously indexed documents have been preserved. Partial ingestion is not complete. "
                "Pass --allow-partial-ingestion to accept a partial index."
            )

    print("\nIngestion complete.")
    print(f"  Store resource        : {summary['store_name']}")
    print(f"  Display name          : {summary['display_name']}")
    print(f"  Documents added/updated: {summary['uploaded']}")
    print(f"  Documents skipped     : {summary['skipped']}")
    print(f"  Documents removed     : {summary['removed']}")
    print("\nThe store is ready. You can now run queries.")
    return summary


def main(
    create_store: bool = False,
    replace: bool = False,
    dry_run: bool = False,
    allow_partial_ingestion: bool = False,
    force_rebuild: bool = False,
    allow_empty: bool = False,
) -> None:
    try:
        ingest_corpus(
            create_store=create_store,
            replace=replace,
            dry_run=dry_run,
            allow_partial_ingestion=allow_partial_ingestion,
            force_rebuild=force_rebuild,
            allow_empty=allow_empty,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize local corpus with a Gemini File Search store safely."
    )
    parser.add_argument(
        "--create-store",
        action="store_true",
        help="Explicitly create a new File Search store for this project if not already configured.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing documents (default is non-destructive sync without deleting).",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Explicitly ensure no remote documents are replaced (default behavior).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect local documents against the store and report planned actions without modifying anything.",
    )
    parser.add_argument(
        "--allow-partial-ingestion",
        action="store_true",
        help="Proceed and accept store even if some files fail to upload.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Deprecated alias for --replace.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Deprecated; an empty corpus is always refused.",
    )
    args = parser.parse_args()

    effective_replace = (args.replace or args.force_rebuild) and not args.no_replace

    main(
        create_store=args.create_store,
        replace=effective_replace,
        dry_run=args.dry_run,
        allow_partial_ingestion=args.allow_partial_ingestion,
        force_rebuild=args.force_rebuild,
        allow_empty=args.allow_empty,
    )
