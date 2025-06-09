"""
Unit tests for the FileTransfer module.
"""

import os
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call
import pytest

from src.transfer import (
    FileTransfer, FileTransferError, InsufficientSpaceError, 
    DeviceDisconnectedError, TransferRecord, AudioFile
)
from src.config import ConfigManager


class TestFileTransfer:
    """Test suite for FileTransfer class."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        source_dir = tempfile.mkdtemp(prefix="test_source_")
        dest_dir = tempfile.mkdtemp(prefix="test_dest_")
        db_dir = tempfile.mkdtemp(prefix="test_db_")
        
        yield {
            'source': Path(source_dir),
            'dest': Path(dest_dir),
            'db': Path(db_dir)
        }
        
        # Cleanup
        shutil.rmtree(source_dir)
        shutil.rmtree(dest_dir)
        shutil.rmtree(db_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dirs):
        """Create a mock ConfigManager."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            "paths.audio_destination": str(temp_dirs['dest']),
            "transfer.supported_formats": ["mp3", "wav", "m4a", "wma"],
            "transfer.min_file_size": 1024,
            "transfer.max_file_size": 2147483648,
            "transfer.cleanup_device": False,
            "transfer.verify_transfer": True,
            "history.database_path": str(temp_dirs['db'] / "history.db"),
            "history.retention_days": 365
        }.get(key, default)
        return config
    
    @pytest.fixture
    def file_transfer(self, mock_config):
        """Create a FileTransfer instance."""
        return FileTransfer(mock_config)
    
    @pytest.fixture
    def sample_audio_files(self, temp_dirs):
        """Create sample audio files for testing."""
        files = []
        
        # Create valid audio files with headers
        audio_data = {
            'test1.mp3': b'ID3\x03\x00\x00\x00' + b'x' * 2048,  # ID3v2 header
            'test2.wav': b'RIFF' + b'\x00' * 4 + b'WAVE' + b'x' * 2048,  # WAV header
            'test3.m4a': b'\x00\x00\x00\x20ftypM4A ' + b'x' * 2048,  # M4A header
            'test4.wma': b'\x30\x26\xb2\x75' + b'x' * 2048,  # WMA header
        }
        
        for filename, data in audio_data.items():
            file_path = temp_dirs['source'] / filename
            file_path.write_bytes(data)
            files.append(file_path)
        
        # Create a subfolder with more files
        subfolder = temp_dirs['source'] / 'subfolder'
        subfolder.mkdir()
        sub_file = subfolder / 'nested.mp3'
        sub_file.write_bytes(b'ID3\x03\x00\x00\x00' + b'y' * 1024)
        files.append(sub_file)
        
        # Create files to be filtered out
        small_file = temp_dirs['source'] / 'too_small.mp3'
        small_file.write_bytes(b'ID3' + b'x' * 10)  # Too small
        
        wrong_ext = temp_dirs['source'] / 'document.txt'
        wrong_ext.write_text("Not an audio file")
        
        return files
    
    def test_init(self, file_transfer, temp_dirs):
        """Test FileTransfer initialization."""
        assert file_transfer.destination_path == temp_dirs['dest']
        assert file_transfer.db_path.exists()
        assert file_transfer.supported_formats == ["mp3", "wav", "m4a", "wma"]
        assert file_transfer.min_file_size == 1024
        assert file_transfer.max_file_size == 2147483648
        
        # Check database was initialized
        conn = sqlite3.connect(file_transfer.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transfers'")
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_get_audio_files(self, file_transfer, sample_audio_files, temp_dirs):
        """Test scanning for audio files."""
        files = file_transfer.get_audio_files(temp_dirs['source'])
        
        # Should find all valid audio files
        assert len(files) == 5  # 4 in root + 1 in subfolder
        
        # Check files are sorted by modification time
        file_names = [f.name for f in files]
        assert 'test1.mp3' in file_names
        assert 'test2.wav' in file_names
        assert 'test3.m4a' in file_names
        assert 'test4.wma' in file_names
        assert 'nested.mp3' in file_names
        
        # Should not include non-audio or too-small files
        assert 'too_small.mp3' not in file_names
        assert 'document.txt' not in file_names
    
    def test_validate_audio_file(self, file_transfer, sample_audio_files):
        """Test audio file validation."""
        # Valid files
        assert file_transfer.validate_audio_file(sample_audio_files[0])  # MP3
        assert file_transfer.validate_audio_file(sample_audio_files[1])  # WAV
        assert file_transfer.validate_audio_file(sample_audio_files[2])  # M4A
        assert file_transfer.validate_audio_file(sample_audio_files[3])  # WMA
        
        # Invalid files
        text_file = sample_audio_files[0].parent / 'test.txt'
        text_file.write_text("Not audio")
        assert not file_transfer.validate_audio_file(text_file)
        
        # Non-existent file
        assert not file_transfer.validate_audio_file(Path('/non/existent/file.mp3'))
    
    def test_calculate_checksum(self, file_transfer, sample_audio_files):
        """Test checksum calculation."""
        checksum1 = file_transfer._calculate_checksum(sample_audio_files[0])
        checksum2 = file_transfer._calculate_checksum(sample_audio_files[0])
        
        # Same file should have same checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex string
        
        # Different files should have different checksums
        checksum3 = file_transfer._calculate_checksum(sample_audio_files[1])
        assert checksum1 != checksum3
    
    def test_is_duplicate_new_file(self, file_transfer, sample_audio_files):
        """Test duplicate detection for new files."""
        # First time should not be duplicate
        assert not file_transfer.is_duplicate(sample_audio_files[0])
    
    def test_is_duplicate_existing_file(self, file_transfer, sample_audio_files):
        """Test duplicate detection for already transferred files."""
        # Transfer a file first
        audio_file = file_transfer._transfer_file(sample_audio_files[0])
        file_transfer._record_transfer(audio_file)
        
        # Now it should be detected as duplicate
        assert file_transfer.is_duplicate(sample_audio_files[0])
        
        # File with same content but different name should not be detected as duplicate
        # unless it has same size (our quick check filters by filename+size first)
        duplicate = sample_audio_files[0].parent / 'copy.mp3'
        shutil.copy2(sample_audio_files[0], duplicate)
        # This won't be detected as duplicate because filename is different
        assert not file_transfer.is_duplicate(duplicate)
    
    def test_transfer_file(self, file_transfer, sample_audio_files, temp_dirs):
        """Test single file transfer."""
        source_file = sample_audio_files[0]
        audio_file = file_transfer._transfer_file(source_file)
        
        # Check transfer result
        assert isinstance(audio_file, AudioFile)
        assert audio_file.source_path == source_file
        assert audio_file.destination_path.exists()
        assert audio_file.destination_path.parent.name == datetime.now().strftime("%Y-%m-%d")
        assert audio_file.filename == source_file.name
        assert audio_file.size == source_file.stat().st_size
        assert audio_file.checksum is not None
        assert audio_file.format == "mp3"
        
        # Verify file was copied correctly
        assert audio_file.destination_path.read_bytes() == source_file.read_bytes()
    
    def test_transfer_file_with_conflict(self, file_transfer, sample_audio_files, temp_dirs):
        """Test file transfer with naming conflict."""
        source_file = sample_audio_files[0]
        
        # Transfer once
        audio_file1 = file_transfer._transfer_file(source_file)
        
        # Transfer again - should create numbered version
        audio_file2 = file_transfer._transfer_file(source_file)
        
        assert audio_file1.destination_path != audio_file2.destination_path
        assert "test1_1.mp3" in str(audio_file2.destination_path)
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk_usage, file_transfer):
        """Test disk space check with sufficient space."""
        mock_disk_usage.return_value = MagicMock(free=1024 * 1024 * 1024)  # 1GB free
        assert file_transfer._check_disk_space(100 * 1024 * 1024)  # Need 100MB
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_disk_usage, file_transfer):
        """Test disk space check with insufficient space."""
        mock_disk_usage.return_value = MagicMock(free=50 * 1024 * 1024)  # 50MB free
        assert not file_transfer._check_disk_space(100 * 1024 * 1024)  # Need 100MB
    
    def test_transfer_new_files_success(self, file_transfer, sample_audio_files, temp_dirs):
        """Test successful batch transfer."""
        transferred = file_transfer.transfer_new_files(temp_dirs['source'])
        
        # Should transfer all valid files
        assert len(transferred) == 5
        
        # Check all files were transferred
        for audio_file in transferred:
            assert audio_file.destination_path.exists()
            assert audio_file.source_path in sample_audio_files
        
        # Running again should transfer nothing (duplicates)
        transferred2 = file_transfer.transfer_new_files(temp_dirs['source'])
        assert len(transferred2) == 0
    
    def test_transfer_new_files_device_disconnected(self, file_transfer):
        """Test transfer with disconnected device."""
        with pytest.raises(DeviceDisconnectedError):
            file_transfer.transfer_new_files(Path("/non/existent/path"))
    
    @patch('shutil.disk_usage')
    def test_transfer_new_files_insufficient_space(self, mock_disk_usage, file_transfer, sample_audio_files, temp_dirs):
        """Test transfer with insufficient disk space."""
        mock_disk_usage.return_value = MagicMock(free=1024)  # Only 1KB free
        
        with pytest.raises(InsufficientSpaceError):
            file_transfer.transfer_new_files(temp_dirs['source'])
    
    def test_record_transfer(self, file_transfer, sample_audio_files):
        """Test recording transfer in database."""
        # Create a transfer record
        audio_file = AudioFile(
            source_path=sample_audio_files[0],
            destination_path=Path("/dest/test.mp3"),
            filename="test.mp3",
            size=1024,
            checksum="abc123",
            timestamp=datetime.now(),
            duration=60.0,
            format="mp3"
        )
        
        file_transfer._record_transfer(audio_file)
        
        # Check database
        conn = sqlite3.connect(file_transfer.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transfers WHERE checksum = ?", ("abc123",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[2] == "test.mp3"  # filename
        assert row[3] == "abc123"  # checksum
    
    def test_get_transfer_history(self, file_transfer):
        """Test retrieving transfer history."""
        # Record some transfers
        for i in range(3):
            audio_file = AudioFile(
                source_path=Path(f"/source/test{i}.mp3"),
                destination_path=Path(f"/dest/test{i}.mp3"),
                filename=f"test{i}.mp3",
                size=1024 * (i + 1),
                checksum=f"checksum{i}",
                timestamp=datetime.now() - timedelta(days=i),
                duration=60.0 * (i + 1),
                format="mp3"
            )
            file_transfer._record_transfer(audio_file)
        
        # Get history
        history = file_transfer.get_transfer_history(days=7)
        
        assert len(history) == 3
        assert all(isinstance(record, TransferRecord) for record in history)
        
        # Should be sorted by date descending
        assert history[0].filename == "test0.mp3"
        assert history[2].filename == "test2.mp3"
    
    def test_cleanup_device(self, file_transfer, sample_audio_files):
        """Test device cleanup functionality."""
        # Create list of paths to clean
        to_clean = sample_audio_files[:2]
        
        # Ensure files exist
        assert all(f.exists() for f in to_clean)
        
        # Temporarily enable cleanup by modifying the boolean attribute
        original_cleanup = file_transfer.cleanup_device
        file_transfer.cleanup_device = True
        
        try:
            # Use the unbound method to avoid the naming conflict
            FileTransfer.cleanup_device(file_transfer, None, to_clean)
            
            # Files should be deleted
            assert not any(f.exists() for f in to_clean)
            
            # Other files should remain
            assert sample_audio_files[2].exists()
        finally:
            # Restore original value
            file_transfer.cleanup_device = original_cleanup
    
    def test_cleanup_old_records(self, file_transfer):
        """Test cleaning up old database records."""
        # Insert old record
        old_date = datetime.now() - timedelta(days=400)
        
        conn = sqlite3.connect(file_transfer.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transfers 
            (file_id, filename, checksum, transfer_date, file_size)
            VALUES (?, ?, ?, ?, ?)
        """, ("old_file", "old.mp3", "old_checksum", old_date, 1024))
        conn.commit()
        conn.close()
        
        # Clean up
        file_transfer._cleanup_old_records()
        
        # Old record should be gone
        conn = sqlite3.connect(file_transfer.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transfers WHERE file_id = ?", ("old_file",))
        assert cursor.fetchone() is None
        conn.close()
    
    def test_get_statistics(self, file_transfer):
        """Test getting transfer statistics."""
        # Record some transfers
        formats = ["mp3", "wav", "mp3", "m4a"]
        for i, fmt in enumerate(formats):
            audio_file = AudioFile(
                source_path=Path(f"/source/test{i}.{fmt}"),
                destination_path=Path(f"/dest/test{i}.{fmt}"),
                filename=f"test{i}.{fmt}",
                size=1024 * 1024 * (i + 1),  # 1MB, 2MB, 3MB, 4MB
                checksum=f"checksum{i}",
                timestamp=datetime.now(),
                duration=60.0,
                format=fmt
            )
            file_transfer._record_transfer(audio_file)
        
        stats = file_transfer.get_statistics()
        
        assert stats['total_transfers'] == 4
        assert stats['total_size_mb'] == 10.0  # 1+2+3+4 MB
        assert stats['formats']['mp3'] == 2
        assert stats['formats']['wav'] == 1
        assert stats['formats']['m4a'] == 1
    
    def test_transfer_with_verification_failure(self, file_transfer, sample_audio_files):
        """Test transfer with checksum verification failure."""
        source_file = sample_audio_files[0]
        
        # Mock checksum to simulate verification failure
        with patch.object(file_transfer, '_calculate_checksum') as mock_checksum:
            mock_checksum.side_effect = ["checksum1", "checksum2"]  # Different checksums
            
            with pytest.raises(FileTransferError) as exc_info:
                file_transfer._transfer_file(source_file)
            
            assert "Checksum verification failed" in str(exc_info.value)
    
    def test_transfer_with_copy_failure(self, file_transfer, sample_audio_files):
        """Test transfer with copy failure."""
        source_file = sample_audio_files[0]
        
        with patch('shutil.copy2', side_effect=IOError("Copy failed")):
            with pytest.raises(FileTransferError):
                file_transfer._transfer_file(source_file)
    
    def test_concurrent_database_access(self, file_transfer):
        """Test thread-safe database access."""
        import threading
        
        def record_transfer(i):
            audio_file = AudioFile(
                source_path=Path(f"/source/test{i}.mp3"),
                destination_path=Path(f"/dest/test{i}.mp3"),
                filename=f"test{i}.mp3",
                size=1024,
                checksum=f"checksum{i}",
                timestamp=datetime.now(),
                duration=60.0,
                format="mp3"
            )
            file_transfer._record_transfer(audio_file)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=record_transfer, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check all records were created
        conn = sqlite3.connect(file_transfer.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transfers")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 10