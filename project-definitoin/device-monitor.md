# Device Monitor Module

## Purpose
Monitors USB device connections and detects when the Sony ICD-TX660 recorder is connected. Provides cross-platform USB device detection for both Linux and macOS.

## Dependencies
- **External**: 
  - Linux: `pyudev`, `subprocess`
  - macOS: `pyobjc`, `subprocess`
  - Common: `threading`, `time`
- **Internal**: [[ config_manager ]]

## Interface

### Input
- Device configuration from [[ config_manager ]]
- USB device events from operating system

### Output
- Device connection events
- Device information (mount path, device ID)

### Public Methods

```python
class DeviceMonitor:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def start_monitoring(self, callback: Callable[[DeviceEvent], None]):
        """Start monitoring for device connections"""
        
    def stop_monitoring(self):
        """Stop monitoring for device connections"""
        
    def get_device_info(self, device_path: str) -> DeviceInfo:
        """Get detailed information about connected device"""
        
    def is_target_device(self, device: Any) -> bool:
        """Check if device matches target configuration"""
        
    @property
    def is_monitoring(self) -> bool:
        """Return current monitoring status"""
```

## Data Structures

```python
@dataclass
class DeviceEvent:
    event_type: str  # "connected" or "disconnected"
    device_path: str
    mount_point: Optional[str]
    timestamp: datetime
    device_info: DeviceInfo

@dataclass
class DeviceInfo:
    vendor_id: str
    product_id: str
    serial_number: str
    device_name: str
    mount_point: str
    total_space: int
    free_space: int
```

## Platform-Specific Implementation

### Linux Implementation
```python
def _monitor_linux(self):
    """Linux-specific USB monitoring using udev"""
    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='usb')
    
    for device in iter(monitor.poll, None):
        if device.action == 'add':
            self._handle_device_connected(device)
```

### macOS Implementation
```python
def _monitor_macos(self):
    """macOS-specific USB monitoring using DiskArbitration"""
    # Use subprocess to monitor diskutil
    # Or use pyobjc for native integration
```

## Event Flow

1. Operating system detects USB device connection
2. Platform-specific monitor receives event
3. Module checks if device matches target configuration
4. If match found, gather device information
5. Wait for device mounting (with timeout)
6. Emit DeviceEvent to registered callback
7. [[ main_controller ]] receives event and initiates pipeline

## Configuration Requirements

```yaml
device:
  identifier: "Sony_ICD-TX660"  # Device name to match
  vendor_id: "054c"            # Sony vendor ID (optional)
  product_id: "0xxx"           # Product ID (optional)
  serial_pattern: "^[A-Z0-9]{8}$"  # Serial number pattern (optional)
```

## Error Handling

### Common Issues
- **Permission Denied**: Need appropriate permissions for USB monitoring
- **Device Not Mounting**: Timeout waiting for mount point
- **Multiple Devices**: Handle multiple matching devices
- **Rapid Connect/Disconnect**: Debounce rapid events

### Recovery Strategies
1. Retry device detection after permission errors
2. Implement exponential backoff for mounting timeout
3. Queue events to handle rapid connections
4. Log all device events for debugging

## Module Relationships
- Uses: [[ config_manager ]]
- Triggers: [[ file_transfer ]] (via [[ main_controller ]])
- Subscribes to: Operating system USB events

## Implementation Notes

1. **Thread Safety**: Monitor runs in separate thread
2. **Event Debouncing**: Prevent duplicate events within 5 seconds
3. **Resource Cleanup**: Properly release system resources on stop
4. **Cross-Platform**: Abstract platform differences in base class
5. **Hot-Plug Support**: Handle devices already connected at startup

## Testing Considerations

- Mock USB device events for unit testing
- Test with multiple device connections
- Test rapid connect/disconnect scenarios
- Verify cross-platform compatibility
- Test with insufficient permissions

## Security Considerations

- Validate device identity beyond just name
- Don't auto-mount untrusted devices
- Log all device connection attempts
- Implement device whitelist/blacklist

## Future Enhancements

- Support for multiple recorder models
- Bluetooth device support
- Network-attached recorder support
- Device health monitoring
- Automatic device firmware updates