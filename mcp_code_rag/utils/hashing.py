"""Hashing utilities for file and string content."""

import hashlib
from pathlib import Path


def hash_file(path: str) -> str:
    """
    Compute SHA-256 hash of a file in 64 KB blocks.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read.
    """
    sha256_hash = hashlib.sha256()
    chunk_size = 64 * 1024  # 64 KB blocks

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def hash_string(content: str) -> str:
    """
    Compute SHA-256 hash of a string.

    Args:
        content: String content to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    """
    Generate a short hash for stable workspace IDs.

    Args:
        value: String to hash.
        length: Desired length of the short hash (default 12).

    Returns:
        Hexadecimal hash string truncated to specified length.

    Raises:
        ValueError: If length is not positive.
    """
    if length <= 0:
        raise ValueError("length must be positive")

    full_hash = hash_string(value)
    return full_hash[:length]
