"""
File Transfer Module for TranscriMatic

Handles transferring audio files from connected devices to local storage
with duplicate detection and transfer history management.
"""

import os
import shutil
import hashlib
import sqlite3
import errno
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass
import threading

from src.config import ConfigManager
from .data_classes import AudioFile


logger = logging.getLogger(__name__)


class FileTransferError(Exception):
    """Base exception for file transfer errors."""
    pass


class InsufficientSpaceError(FileTransferError):
    """Raised when there's not enough disk space."""
    pass


class DeviceDisconnectedError(FileTransferError):
    """Raised when device is disconnected during transfer."""
    pass


@dataclass
class TransferRecord:
    """Record of a file transfer in the history database."""
    file_id: str
    filename: str
    checksum: str
    transfer_date: datetime
    source_device: str
    file_size: int
    source_path: Optional[str] = None
    destination_path: Optional[str] = None
    duration: Optional[float] = None
    format: Optional[str] = None


class FileTransfer:
    """
    Manages file transfers from audio recording devices to local storage.
    Handles duplicate detection, transfer verification, and history tracking.
    """
    
    # Default supported audio formats
    DEFAULT_FORMATS = ['mp3', 'wav', 'm4a', 'wma', 'opus', 'flac', 'ogg']
    
    # File size limits
    DEFAULT_MIN_SIZE = 1024  # 1KB
    DEFAULT_MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Chunk size for checksum calculation
    CHUNK_SIZE = 8192
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize FileTransfer with configuration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        
        # Load configuration
        self.destination_path = Path(self.config.get("paths.audio_destination"))
        self.supported_formats = self.config.get("transfer.supported_formats", self.DEFAULT_FORMATS)
        self.min_file_size = self.config.get("transfer.min_file_size", self.DEFAULT_MIN_SIZE)
        self.max_file_size = self.config.get("transfer.max_file_size", self.DEFAULT_MAX_SIZE)
        self.cleanup_device = self.config.get("transfer.cleanup_device", False)
        self.verify_transfer = self.config.get("transfer.verify_transfer", True)
        
        # Database configuration
        db_path = self.config.get("history.database_path", "~/.transcrimatic/history.db")
        self.db_path = Path(os.path.expanduser(db_path))
        self.retention_days = self.config.get("history.retention_days", 365)
        
        # Thread lock for database access
        self._db_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Ensure destination directory exists
        self.destination_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileTransfer initialized with destination: {self.destination_path}")
    
    def transfer_new_files(self, source_path: Path) -> List[AudioFile]:
        """
        Transfer new audio files from source to destination.
        
        Args:
            source_path: Path to source device/directory
            
        Returns:
            List of successfully transferred AudioFile objects
            
        Raises:
            FileTransferError: If transfer fails
            InsufficientSpaceError: If not enough disk space
            DeviceDisconnectedError: If device disconnected
        """
        transferred = []
        
        # Verify source exists
        if not source_path.exists():
            raise DeviceDisconnectedError(f"Source path not found: {source_path}")
        
        try:
            # 1. Scan for audio files
            audio_files = self.get_audio_files(source_path)
            logger.info(f"Found {len(audio_files)} audio files on device")
            
            # 2. Filter already transferred
            new_files = []
            for file_path in audio_files:
                if not self.is_duplicate(file_path):
                    new_files.append(file_path)
                else:
                    logger.debug(f"Skipping duplicate: {file_path.name}")
            
            logger.info(f"{len(new_files)} new files to transfer")
            
            if not new_files:
                return transferred
            
            # 3. Check destination space
            required_space = sum(f.stat().st_size for f in new_files)
            if not self._check_disk_space(required_space):
                raise InsufficientSpaceError(
                    f"Need {required_space / 1024 / 1024:.1f}MB but insufficient space available"
                )
            
            # 4. Transfer each file
            for i, file_path in enumerate(new_files, 1):
                try:
                    logger.info(f"Transferring file {i}/{len(new_files)}: {file_path.name}")
                    
                    if self.validate_audio_file(file_path):
                        audio_file = self._transfer_file(file_path)
                        transferred.append(audio_file)
                        self._record_transfer(audio_file)
                        logger.info(f"Successfully transferred: {file_path.name}")
                    else:
                        logger.warning(f"Invalid audio file skipped: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to transfer {file_path}: {e}")
                    continue
            
            # 5. Optional cleanup
            if self.cleanup_device and transferred:
                self.cleanup_device(source_path, [f.source_path for f in transferred])
            
            logger.info(f"Transfer complete: {len(transferred)} files transferred")
            return transferred
            
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            raise
    
    def get_audio_files(self, device_path: Path) -> List[Path]:
        """
        Scan device for audio files.
        
        Args:
            device_path: Root path to scan
            
        Returns:
            List of audio file paths sorted by modification time
        """
        audio_files = []
        
        try:
            # Recursively find audio files
            for ext in self.supported_formats:
                # Case-insensitive search
                for pattern in [f"*.{ext}", f"*.{ext.upper()}"]:
                    audio_files.extend(device_path.rglob(pattern))
            
            # Filter by size
            valid_files = []
            for file_path in audio_files:
                try:
                    size = file_path.stat().st_size
                    if self.min_file_size <= size <= self.max_file_size:
                        valid_files.append(file_path)
                    else:
                        logger.debug(f"File size out of range: {file_path} ({size} bytes)")
                except Exception as e:
                    logger.warning(f"Cannot stat file {file_path}: {e}")
            
            # Sort by modification time (oldest first)
            valid_files.sort(key=lambda f: f.stat().st_mtime)
            
            return valid_files
            
        except Exception as e:
            logger.error(f"Error scanning device: {e}")
            return []
    
    def is_duplicate(self, file_path: Path) -> bool:
        """
        Check if file has already been transferred.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file is duplicate, False otherwise
        """
        try:
            # Quick check: filename + size
            file_stat = file_path.stat()
            filename = file_path.name
            size = file_stat.st_size
            
            # Check database for same filename and size
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT checksum FROM transfers 
                    WHERE filename = ? AND file_size = ?
                """, (filename, size))
                
                existing_checksums = [row[0] for row in cursor.fetchall()]
                conn.close()
            
            if not existing_checksums:
                return False
            
            # Calculate checksum for comparison
            file_checksum = self._calculate_checksum(file_path)
            
            return file_checksum in existing_checksums
            
        except Exception as e:
            logger.error(f"Error checking duplicate for {file_path}: {e}")
            return False
    
    def validate_audio_file(self, file_path: Path) -> bool:
        """
        Validate audio file format and integrity.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if valid audio file, False otherwise
        """
        try:
            # Check file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check extension
            ext = file_path.suffix[1:].lower()
            if ext not in self.supported_formats:
                return False
            
            # Check file size
            size = file_path.stat().st_size
            if size < self.min_file_size or size > self.max_file_size:
                return False
            
            # Basic header validation for common formats
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
                # MP3: Check for ID3 tag or MPEG sync
                if ext == 'mp3':
                    if header.startswith(b'ID3') or (header[0:2] == b'\xff\xfb'):
                        return True
                
                # WAV: RIFF header
                elif ext == 'wav':
                    if header.startswith(b'RIFF') and b'WAVE' in header:
                        return True
                
                # M4A/MP4: ftyp box
                elif ext in ['m4a', 'mp4']:
                    if b'ftyp' in header:
                        return True
                
                # WMA: ASF header
                elif ext == 'wma':
                    if header.startswith(b'\x30\x26\xb2\x75'):
                        return True
                
                # FLAC: fLaC marker
                elif ext == 'flac':
                    if header.startswith(b'fLaC'):
                        return True
                
                # OGG: OggS header
                elif ext in ['ogg', 'opus']:
                    if header.startswith(b'OggS'):
                        return True
                
                # If no specific validation, accept based on extension
                else:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False
    
    def cleanup_device(self, device_path: Path, transferred_files: List[Path]):
        """
        Remove transferred files from device.
        
        Args:
            device_path: Device root path
            transferred_files: List of file paths to remove
        """
        if not self.cleanup_device:
            return
        
        logger.info(f"Cleaning up {len(transferred_files)} transferred files from device")
        
        for file_path in transferred_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Deleted from device: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
    
    def get_transfer_history(self, days: int = 30) -> List[TransferRecord]:
        """
        Get recent transfer history.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of TransferRecord objects
        """
        records = []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_id, filename, checksum, transfer_date, 
                           device_serial, file_size, source_path, destination_path,
                           duration, format
                    FROM transfers 
                    WHERE transfer_date > ?
                    ORDER BY transfer_date DESC
                """, (cutoff_date,))
                
                for row in cursor.fetchall():
                    records.append(TransferRecord(
                        file_id=row['file_id'],
                        filename=row['filename'],
                        checksum=row['checksum'],
                        transfer_date=datetime.fromisoformat(row['transfer_date']),
                        source_device=row['device_serial'] or 'unknown',
                        file_size=row['file_size'],
                        source_path=row['source_path'],
                        destination_path=row['destination_path'],
                        duration=row['duration'],
                        format=row['format']
                    ))
                
                conn.close()
            
        except Exception as e:
            logger.error(f"Error retrieving transfer history: {e}")
        
        return records
    
    def _init_database(self):
        """Initialize the transfer history database."""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create transfers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transfers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT UNIQUE NOT NULL,
                        filename TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        source_path TEXT,
                        destination_path TEXT,
                        transfer_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        device_serial TEXT,
                        duration REAL,
                        format TEXT
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_checksum ON transfers(checksum)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfer_date ON transfers(transfer_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON transfers(filename)")
                
                conn.commit()
                conn.close()
            
            # Clean up old records
            self._cleanup_old_records()
            
            logger.debug(f"Database initialized at: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex string of checksum
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(self.CHUNK_SIZE), b''):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            raise
    
    def _transfer_file(self, source_path: Path) -> AudioFile:
        """
        Transfer a single file with verification.
        
        Args:
            source_path: Source file path
            
        Returns:
            AudioFile object with transfer details
            
        Raises:
            FileTransferError: If transfer fails
        """
        # Generate destination path with date organization
        date_dir = datetime.now().strftime("%Y-%m-%d")
        dest_dir = self.destination_path / date_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle filename conflicts
        dest_path = dest_dir / source_path.name
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            # Copy file with metadata preservation
            temp_path = dest_path.with_suffix('.tmp')
            shutil.copy2(source_path, temp_path)
            
            # Verify transfer if enabled
            if self.verify_transfer:
                source_checksum = self._calculate_checksum(source_path)
                temp_checksum = self._calculate_checksum(temp_path)
                
                if source_checksum != temp_checksum:
                    temp_path.unlink()
                    raise FileTransferError("Checksum verification failed")
            else:
                source_checksum = self._calculate_checksum(source_path)
            
            # Atomic rename
            temp_path.rename(dest_path)
            
            # Get file info
            file_stat = source_path.stat()
            
            # Extract audio duration if possible
            duration = self._get_audio_duration(dest_path)
            
            return AudioFile(
                source_path=source_path,
                destination_path=dest_path,
                filename=source_path.name,
                size=file_stat.st_size,
                checksum=source_checksum,
                timestamp=datetime.now(),
                duration=duration,
                format=source_path.suffix[1:].lower()
            )
            
        except Exception as e:
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            if dest_path.exists() and not source_path.exists():
                dest_path.unlink()
            
            raise FileTransferError(f"Transfer failed: {e}")
    
    def _record_transfer(self, audio_file: AudioFile):
        """Record successful transfer in database."""
        try:
            file_id = f"{audio_file.checksum}_{audio_file.timestamp.timestamp()}"
            
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO transfers 
                    (file_id, filename, checksum, source_path, destination_path,
                     transfer_date, file_size, duration, format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_id,
                    audio_file.filename,
                    audio_file.checksum,
                    str(audio_file.source_path),
                    str(audio_file.destination_path),
                    audio_file.timestamp,
                    audio_file.size,
                    audio_file.duration,
                    audio_file.format
                ))
                
                conn.commit()
                conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record transfer: {e}")
    
    def _check_disk_space(self, required_bytes: int) -> bool:
        """Check if destination has enough free space."""
        try:
            stat = shutil.disk_usage(self.destination_path)
            # Keep 100MB buffer
            available = stat.free - (100 * 1024 * 1024)
            return available >= required_bytes
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return True  # Optimistic approach
    
    def _cleanup_old_records(self):
        """Remove old records from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM transfers 
                    WHERE transfer_date < ?
                """, (cutoff_date,))
                
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old transfer records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
    
    def _get_audio_duration(self, file_path: Path) -> Optional[float]:
        """
        Try to extract audio duration from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds or None if cannot determine
        """
        try:
            # This is a placeholder - in production you'd use mutagen or similar
            # For now, return None and let the audio processor determine duration
            return None
        except Exception:
            return None
    
    def get_statistics(self) -> dict:
        """Get transfer statistics."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Total transfers
                cursor.execute("SELECT COUNT(*) FROM transfers")
                total_transfers = cursor.fetchone()[0]
                
                # Total size
                cursor.execute("SELECT SUM(file_size) FROM transfers")
                total_size = cursor.fetchone()[0] or 0
                
                # Transfers by format
                cursor.execute("""
                    SELECT format, COUNT(*) 
                    FROM transfers 
                    GROUP BY format
                """)
                format_stats = dict(cursor.fetchall())
                
                conn.close()
            
            return {
                'total_transfers': total_transfers,
                'total_size_mb': total_size / 1024 / 1024,
                'formats': format_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}