# File Transfer Module

## Purpose
Handles the transfer of audio files from the connected Sony recorder to the local file system. Manages duplicate detection, file validation, and transfer history to ensure only new recordings are processed.

## Dependencies
- **External**: `shutil`, `hashlib`, `sqlite3`, `pathlib`
- **Internal**: [[ config_manager ]]

## Interface

### Input
- Device mount point from [[ device_monitor ]]
- Destination path from [[ config_manager ]]

### Output
- List of successfully transferred audio files
- Transfer statistics and history

### Public Methods

```python
class FileTransfer:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def transfer_new_files(self, source_path: Path) -> List[AudioFile]:
        """Transfer only new files from device"""
        
    def get_audio_files(self, device_path: Path) -> List[Path]:
        """Scan device for audio files"""
        
    def is_duplicate(self, file_path: Path) -> bool:
        """Check if file has already been transferred"""
        
    def validate_audio_file(self, file_path: Path) -> bool:
        """Validate audio file format and integrity"""
        
    def cleanup_device(self, device_path: Path, transferred_files: List[Path]):
        """Optional: Remove transferred files from device"""
        
    def get_transfer_history(self, days: int = 30) -> List[TransferRecord]:
        """Get recent transfer history"""
```

## Data Structures

```python
@dataclass
class AudioFile:
    source_path: Path
    destination_path: Path
    filename: str
    size: int
    checksum: str
    timestamp: datetime
    duration: Optional[float]
    format: str  # mp3, wav, etc.

@dataclass
class TransferRecord:
    file_id: str
    filename: str
    checksum: str
    transfer_date: datetime
    source_device: str
    file_size: int
```

## Database Schema

```sql
-- transfer_history.db
CREATE TABLE transfers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    checksum TEXT NOT NULL,
    source_path TEXT,
    destination_path TEXT,
    transfer_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    device_serial TEXT,
    INDEX idx_checksum (checksum),
    INDEX idx_transfer_date (transfer_date)
);
```

## Transfer Algorithm

```python
def transfer_new_files(self, source_path: Path) -> List[AudioFile]:
    transferred = []
    
    # 1. Scan for audio files
    audio_files = self.get_audio_files(source_path)
    
    # 2. Filter already transferred
    new_files = [f for f in audio_files if not self.is_duplicate(f)]
    
    # 3. Transfer each file
    for file in new_files:
        if self.validate_audio_file(file):
            dest = self._transfer_file(file)
            transferred.append(dest)
            self._record_transfer(dest)
    
    return transferred
```

## Supported Audio Formats

- **Primary**: MP3, WAV, M4A (Sony recorder defaults)
- **Additional**: FLAC, OGG, WMA
- **Validation**: Check file headers, not just extensions

## Duplicate Detection Strategy

1. **Checksum Comparison**: SHA-256 hash of file content
2. **Filename + Size**: Quick pre-filter before checksum
3. **Timestamp Window**: Consider files within 1-minute window
4. **Database Lookup**: Check transfer history

## Error Handling

### Common Errors
- **Insufficient Space**: Check destination has enough space
- **Read Permission**: Handle protected files on device
- **Corrupted Files**: Skip and log corrupted audio files
- **Database Lock**: Handle concurrent access to history DB

### Recovery Strategies
```python
try:
    # Attempt transfer
    shutil.copy2(source, destination)
except IOError as e:
    if e.errno == errno.ENOSPC:
        # Handle no space
        self._cleanup_old_files()
        # Retry transfer
    elif e.errno == errno.EACCES:
        # Handle permission denied
        self._log_permission_error(source)
    else:
        raise
```

## Performance Optimization

1. **Batch Operations**: Transfer multiple files in single transaction
2. **Parallel Transfers**: Use thread pool for large files
3. **Progress Tracking**: Emit progress events for UI updates
4. **Incremental Scanning**: Cache device file listings

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Triggers: [[ audio_processor ]]
- Outputs to: Configured audio destination folder

## Configuration

```yaml
paths:
  audio_destination: "/home/user/audio_recordings"
  
transfer:
  supported_formats: ["mp3", "wav", "m4a"]
  min_file_size: 1024  # bytes
  max_file_size: 2147483648  # 2GB
  cleanup_device: false
  verify_transfer: true
  
history:
  database_path: "~/.audio_transfer/history.db"
  retention_days: 365
```

## Testing Considerations

- Test with various audio formats
- Simulate device disconnection during transfer
- Test duplicate detection accuracy
- Verify database integrity after crashes
- Test with full destination disk

## Security Considerations

- Sanitize filenames from device
- Verify file sizes before transfer
- Scan for suspicious file patterns
- Implement transfer quotas

## Future Enhancements

- Cloud backup integration
- Compression during transfer
- Metadata preservation
- Resume interrupted transfers
- Smart file organization by date/content