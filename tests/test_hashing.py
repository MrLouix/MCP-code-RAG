"""Tests for hashing utilities."""

import logging
import tempfile
from pathlib import Path

import pytest

from mcp_code_rag.logging_config import setup_logging
from mcp_code_rag.utils.hashing import hash_file, hash_string, short_hash


class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_file_returns_64_char_hex(self):
        """Test that hash_file returns a 64-character hexadecimal string."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()
            tmp_path = tmp.name

        try:
            result = hash_file(tmp_path)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            Path(tmp_path).unlink()

    def test_identical_files_same_hash(self):
        """Test that identical files produce the same hash."""
        content = b"identical content"

        with tempfile.NamedTemporaryFile(delete=False) as tmp1:
            tmp1.write(content)
            tmp1.flush()
            path1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp2:
            tmp2.write(content)
            tmp2.flush()
            path2 = tmp2.name

        try:
            hash1 = hash_file(path1)
            hash2 = hash_file(path2)
            assert hash1 == hash2
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_modified_file_different_hash(self):
        """Test that modifying a file changes its hash."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"original content")
            tmp.flush()
            tmp_path = tmp.name

        try:
            hash1 = hash_file(tmp_path)

            # Modify file
            with open(tmp_path, "wb") as f:
                f.write(b"modified content")

            hash2 = hash_file(tmp_path)
            assert hash1 != hash2
        finally:
            Path(tmp_path).unlink()

    def test_hash_file_nonexistent_raises_error(self):
        """Test that hash_file raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/file/path.txt")


class TestHashString:
    """Tests for hash_string function."""

    def test_hash_string_returns_64_char_hex(self):
        """Test that hash_string returns a 64-character hexadecimal string."""
        result = hash_string("test content")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_identical_strings_same_hash(self):
        """Test that identical strings produce the same hash."""
        content = "identical content"
        hash1 = hash_string(content)
        hash2 = hash_string(content)
        assert hash1 == hash2

    def test_different_strings_different_hash(self):
        """Test that different strings produce different hashes."""
        hash1 = hash_string("content1")
        hash2 = hash_string("content2")
        assert hash1 != hash2


class TestShortHash:
    """Tests for short_hash function."""

    def test_short_hash_exact_length(self):
        """Test that short_hash returns exact requested length."""
        for length in [8, 12, 16, 24]:
            result = short_hash("test value", length=length)
            assert len(result) == length

    def test_short_hash_default_length_12(self):
        """Test that short_hash defaults to length 12."""
        result = short_hash("test value")
        assert len(result) == 12

    def test_short_hash_is_hex(self):
        """Test that short_hash returns hexadecimal characters."""
        result = short_hash("test value", length=12)
        assert all(c in "0123456789abcdef" for c in result)

    def test_short_hash_invalid_length_raises_error(self):
        """Test that short_hash raises ValueError for invalid length."""
        with pytest.raises(ValueError):
            short_hash("test value", length=0)

        with pytest.raises(ValueError):
            short_hash("test value", length=-1)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_valid_levels(self):
        """Test setup_logging with all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            # Should not raise
            setup_logging(level=level)

    def test_setup_logging_valid_formats(self):
        """Test setup_logging with all valid formats."""
        for fmt in ["text", "json"]:
            # Should not raise
            setup_logging(format=fmt)

    def test_setup_logging_invalid_level_raises_error(self):
        """Test setup_logging raises ValueError for invalid level."""
        with pytest.raises(ValueError, match="level must be one of"):
            setup_logging(level="INVALID")

    def test_setup_logging_invalid_format_raises_error(self):
        """Test setup_logging raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="format must be one of"):
            setup_logging(format="invalid")

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(level="INFO", file=str(log_file))

            # Log something
            logger = logging.getLogger("mcp_code_rag")
            logger.info("test message")

            # Verify file was created
            assert log_file.exists()

    def test_setup_logging_creates_parent_directories(self):
        """Test that setup_logging creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "dir" / "test.log"
            setup_logging(level="INFO", file=str(log_file))

            logger = logging.getLogger("mcp_code_rag")
            logger.info("test message")

            assert log_file.exists()
