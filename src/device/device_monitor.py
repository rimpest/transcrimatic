"""
Device Monitor Module for TranscriMatic

This module monitors USB device connections and detects when the target
audio recorder is connected, providing cross-platform support.
"""

import os
import re
import sys
import time
import logging
import threading
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
from collections import deque

# Platform-specific imports
if sys.platform.startswith('linux'):
    try:
        import pyudev
    except ImportError:
        pyudev = None
        logging.warning("pyudev not available - USB monitoring may be limited")
elif sys.platform == 'darwin':
    # macOS implementation using subprocess
    pass

from src.config import ConfigManager


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about a connected USB device."""
    vendor_id: str
    product_id: str
    serial_number: str
    device_name: str
    mount_point: str
    total_space: int
    free_space: int


@dataclass
class DeviceEvent:
    """Event emitted when a device is connected or disconnected."""
    event_type: str  # "connected" or "disconnected"
    device_path: str
    mount_point: Optional[str]
    timestamp: datetime
    device_info: Optional[DeviceInfo]


class DeviceMonitor:
    """
    Monitors USB device connections and detects target audio recorder.
    Provides cross-platform support for Linux and macOS.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the DeviceMonitor with configuration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.callback: Optional[Callable[[DeviceEvent], None]] = None
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Event debouncing
        self._recent_events: deque = deque(maxlen=10)
        self._event_lock = threading.Lock()
        self.debounce_window = 5.0  # seconds
        
        # Device configuration
        self.target_identifier = self.config.get("device.identifier", "Sony_ICD-TX660")
        self.target_vendor_id = self.config.get("device.vendor_id")
        self.target_product_id = self.config.get("device.product_id")
        self.serial_pattern = self.config.get("device.serial_pattern")
        
        # Mount detection settings
        self.mount_timeout = 30  # seconds
        self.mount_check_interval = 0.5  # seconds
        
        logger.info(f"DeviceMonitor initialized for device: {self.target_identifier}")
    
    def start_monitoring(self, callback: Callable[[DeviceEvent], None]):
        """
        Start monitoring for device connections.
        
        Args:
            callback: Function to call when device events occur
        """
        if self.monitoring:
            logger.warning("Device monitoring already active")
            return
        
        self.callback = callback
        self.monitoring = True
        self._stop_event.clear()
        
        # Start monitoring thread based on platform
        if sys.platform.startswith('linux'):
            self.monitor_thread = threading.Thread(
                target=self._monitor_linux,
                daemon=True,
                name="DeviceMonitor"
            )
        elif sys.platform == 'darwin':
            self.monitor_thread = threading.Thread(
                target=self._monitor_macos,
                daemon=True,
                name="DeviceMonitor"
            )
        else:
            raise NotImplementedError(f"Platform {sys.platform} not supported")
        
        self.monitor_thread.start()
        logger.info("Device monitoring started")
        
        # Check for already connected devices
        self._check_connected_devices()
    
    def stop_monitoring(self):
        """Stop monitoring for device connections."""
        if not self.monitoring:
            return
        
        logger.info("Stopping device monitoring...")
        self.monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop cleanly")
        
        self.monitor_thread = None
        self.callback = None
        logger.info("Device monitoring stopped")
    
    @property
    def is_monitoring(self) -> bool:
        """Return current monitoring status."""
        return self.monitoring
    
    def is_target_device(self, device: Any) -> bool:
        """
        Check if device matches target configuration.
        
        Args:
            device: Device object (format depends on platform)
            
        Returns:
            True if device matches target configuration
        """
        try:
            if sys.platform.startswith('linux'):
                return self._is_target_device_linux(device)
            elif sys.platform == 'darwin':
                return self._is_target_device_macos(device)
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking device: {e}")
            return False
    
    def get_device_info(self, device_path: str) -> Optional[DeviceInfo]:
        """
        Get detailed information about connected device.
        
        Args:
            device_path: Path to the device
            
        Returns:
            DeviceInfo object or None if not found
        """
        try:
            if sys.platform.startswith('linux'):
                return self._get_device_info_linux(device_path)
            elif sys.platform == 'darwin':
                return self._get_device_info_macos(device_path)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return None
    
    def _should_emit_event(self, device_path: str, event_type: str) -> bool:
        """
        Check if event should be emitted (debouncing).
        
        Args:
            device_path: Device path
            event_type: Type of event
            
        Returns:
            True if event should be emitted
        """
        with self._event_lock:
            current_time = time.time()
            
            # Clean old events
            while self._recent_events and (current_time - self._recent_events[0][2]) > self.debounce_window:
                self._recent_events.popleft()
            
            # Check for duplicate events
            for path, evt_type, timestamp in self._recent_events:
                if path == device_path and evt_type == event_type:
                    logger.debug(f"Debouncing duplicate event: {event_type} for {device_path}")
                    return False
            
            # Add to recent events
            self._recent_events.append((device_path, event_type, current_time))
            return True
    
    def _emit_event(self, event: DeviceEvent):
        """
        Emit device event to callback.
        
        Args:
            event: DeviceEvent to emit
        """
        if not self.callback:
            logger.warning("No callback registered for device events")
            return
        
        if not self._should_emit_event(event.device_path, event.event_type):
            return
        
        try:
            logger.info(f"Emitting device event: {event.event_type} for {event.device_path}")
            self.callback(event)
        except Exception as e:
            logger.error(f"Error in device event callback: {e}")
    
    def _wait_for_mount(self, device_path: str, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for device to be mounted.
        
        Args:
            device_path: Device path
            timeout: Maximum time to wait (uses self.mount_timeout if None)
            
        Returns:
            Mount point path or None if timeout
        """
        timeout = timeout or self.mount_timeout
        start_time = time.time()
        
        logger.info(f"Waiting for device {device_path} to mount (timeout: {timeout}s)...")
        
        while (time.time() - start_time) < timeout:
            if self._stop_event.is_set():
                return None
            
            mount_point = self._get_mount_point(device_path)
            if mount_point:
                logger.info(f"Device mounted at: {mount_point}")
                return mount_point
            
            time.sleep(self.mount_check_interval)
        
        logger.warning(f"Timeout waiting for device to mount: {device_path}")
        return None
    
    def _get_mount_point(self, device_path: str) -> Optional[str]:
        """
        Get mount point for device.
        
        Args:
            device_path: Device path
            
        Returns:
            Mount point path or None
        """
        if sys.platform.startswith('linux'):
            return self._get_mount_point_linux(device_path)
        elif sys.platform == 'darwin':
            return self._get_mount_point_macos(device_path)
        else:
            return None
    
    def _check_connected_devices(self):
        """Check for already connected devices at startup."""
        logger.info("Checking for already connected devices...")
        
        if sys.platform.startswith('linux'):
            self._check_connected_devices_linux()
        elif sys.platform == 'darwin':
            self._check_connected_devices_macos()
    
    # Linux-specific implementations
    
    def _monitor_linux(self):
        """Linux-specific USB monitoring using udev."""
        if not pyudev:
            logger.error("pyudev not available - cannot monitor USB devices")
            return
        
        try:
            context = pyudev.Context()
            monitor = pyudev.Monitor.from_netlink(context)
            monitor.filter_by(subsystem='usb')
            
            logger.info("Starting Linux USB monitor...")
            
            for device in iter(monitor.poll, None):
                if not self.monitoring or self._stop_event.is_set():
                    break
                
                if device.action == 'add':
                    logger.debug(f"USB device added: {device.device_path}")
                    if self.is_target_device(device):
                        self._handle_device_connected_linux(device)
                elif device.action == 'remove':
                    logger.debug(f"USB device removed: {device.device_path}")
                    # Note: We might want to handle disconnection events
                    
        except Exception as e:
            logger.error(f"Error in Linux USB monitor: {e}")
            self.monitoring = False
    
    def _is_target_device_linux(self, device) -> bool:
        """Check if Linux udev device matches target."""
        try:
            # Get device properties
            vendor_id = device.get('ID_VENDOR_ID', '')
            product_id = device.get('ID_MODEL_ID', '')
            serial = device.get('ID_SERIAL_SHORT', '')
            model = device.get('ID_MODEL', '')
            
            # Check device identifier (model name)
            if self.target_identifier:
                if self.target_identifier not in model:
                    return False
            
            # Check vendor ID if specified
            if self.target_vendor_id:
                if vendor_id.lower() != self.target_vendor_id.lower():
                    return False
            
            # Check product ID if specified
            if self.target_product_id:
                if product_id.lower() != self.target_product_id.lower():
                    return False
            
            # Check serial pattern if specified
            if self.serial_pattern and serial:
                if not re.match(self.serial_pattern, serial):
                    return False
            
            logger.info(f"Target device found: {model} (vendor={vendor_id}, product={product_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Linux device: {e}")
            return False
    
    def _handle_device_connected_linux(self, device):
        """Handle device connection on Linux."""
        try:
            device_path = device.device_path
            
            # Wait for mount
            mount_point = self._wait_for_mount(device_path)
            
            # Get device info
            device_info = self.get_device_info(device_path) if mount_point else None
            
            # Create and emit event
            event = DeviceEvent(
                event_type="connected",
                device_path=device_path,
                mount_point=mount_point,
                timestamp=datetime.now(),
                device_info=device_info
            )
            
            self._emit_event(event)
            
        except Exception as e:
            logger.error(f"Error handling Linux device connection: {e}")
    
    def _get_mount_point_linux(self, device_path: str) -> Optional[str]:
        """Get mount point on Linux."""
        try:
            # Check /proc/mounts
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        device, mount = parts[0], parts[1]
                        # Look for device that might be related to our USB device
                        if self.target_identifier in mount:
                            if os.path.exists(mount) and os.access(mount, os.R_OK):
                                return mount
            
            # Alternative: check media directories
            media_dirs = ['/media', '/mnt', f'/media/{os.environ.get("USER", "")}']
            for media_dir in media_dirs:
                if not os.path.exists(media_dir):
                    continue
                    
                for entry in os.listdir(media_dir):
                    if self.target_identifier in entry:
                        full_path = os.path.join(media_dir, entry)
                        if os.path.ismount(full_path):
                            return full_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Linux mount point: {e}")
            return None
    
    def _get_device_info_linux(self, device_path: str) -> Optional[DeviceInfo]:
        """Get device info on Linux."""
        try:
            # This is a simplified implementation
            # In practice, we'd need to map the device path to actual device info
            mount_point = self._get_mount_point(device_path)
            if not mount_point:
                return None
            
            # Get disk space info
            stat = shutil.disk_usage(mount_point)
            
            return DeviceInfo(
                vendor_id=self.target_vendor_id or "054c",
                product_id=self.target_product_id or "unknown",
                serial_number="unknown",
                device_name=self.target_identifier,
                mount_point=mount_point,
                total_space=stat.total,
                free_space=stat.free
            )
            
        except Exception as e:
            logger.error(f"Error getting Linux device info: {e}")
            return None
    
    def _check_connected_devices_linux(self):
        """Check already connected devices on Linux."""
        if not pyudev:
            return
        
        try:
            context = pyudev.Context()
            for device in context.list_devices(subsystem='usb'):
                if self.is_target_device(device):
                    logger.info("Found already connected target device")
                    self._handle_device_connected_linux(device)
                    break
        except Exception as e:
            logger.error(f"Error checking connected devices on Linux: {e}")
    
    # macOS-specific implementations
    
    def _monitor_macos(self):
        """macOS-specific USB monitoring using diskutil."""
        logger.info("Starting macOS USB monitor...")
        
        # Keep track of known devices
        known_devices = set(self._get_current_devices_macos())
        
        while self.monitoring and not self._stop_event.is_set():
            try:
                current_devices = set(self._get_current_devices_macos())
                
                # Check for new devices
                new_devices = current_devices - known_devices
                for device in new_devices:
                    logger.debug(f"New device detected: {device}")
                    if self._is_target_device_macos(device):
                        self._handle_device_connected_macos(device)
                
                # Update known devices
                known_devices = current_devices
                
                # Sleep before next check
                time.sleep(2.0)  # Poll every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in macOS USB monitor: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _get_current_devices_macos(self) -> List[str]:
        """Get list of current devices on macOS."""
        try:
            result = subprocess.run(
                ['diskutil', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            
            devices = []
            for line in result.stdout.split('\n'):
                if '/dev/disk' in line:
                    parts = line.split()
                    if parts:
                        devices.append(parts[0])
            
            return devices
            
        except Exception as e:
            logger.error(f"Error getting macOS devices: {e}")
            return []
    
    def _is_target_device_macos(self, device_path: str) -> bool:
        """Check if macOS device matches target."""
        try:
            result = subprocess.run(
                ['diskutil', 'info', device_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            info = result.stdout
            
            # Check device name
            if self.target_identifier:
                if self.target_identifier not in info:
                    return False
            
            logger.info(f"Target device found: {device_path}")
            return True
            
        except Exception as e:
            logger.debug(f"Error checking macOS device {device_path}: {e}")
            return False
    
    def _handle_device_connected_macos(self, device_path: str):
        """Handle device connection on macOS."""
        try:
            # Wait for mount
            mount_point = self._wait_for_mount(device_path)
            
            # Get device info
            device_info = self.get_device_info(device_path) if mount_point else None
            
            # Create and emit event
            event = DeviceEvent(
                event_type="connected",
                device_path=device_path,
                mount_point=mount_point,
                timestamp=datetime.now(),
                device_info=device_info
            )
            
            self._emit_event(event)
            
        except Exception as e:
            logger.error(f"Error handling macOS device connection: {e}")
    
    def _get_mount_point_macos(self, device_path: str) -> Optional[str]:
        """Get mount point on macOS."""
        try:
            result = subprocess.run(
                ['diskutil', 'info', device_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if 'Mount Point:' in line:
                    mount_point = line.split(':', 1)[1].strip()
                    if mount_point and os.path.exists(mount_point):
                        return mount_point
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting macOS mount point: {e}")
            return None
    
    def _get_device_info_macos(self, device_path: str) -> Optional[DeviceInfo]:
        """Get device info on macOS."""
        try:
            mount_point = self._get_mount_point(device_path)
            if not mount_point:
                return None
            
            # Get disk space info
            stat = shutil.disk_usage(mount_point)
            
            # Get device info from diskutil
            result = subprocess.run(
                ['diskutil', 'info', device_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            info_dict = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info_dict[key.strip()] = value.strip()
            
            return DeviceInfo(
                vendor_id=self.target_vendor_id or "unknown",
                product_id=self.target_product_id or "unknown",
                serial_number=info_dict.get('Volume UUID', 'unknown'),
                device_name=info_dict.get('Volume Name', self.target_identifier),
                mount_point=mount_point,
                total_space=stat.total,
                free_space=stat.free
            )
            
        except Exception as e:
            logger.error(f"Error getting macOS device info: {e}")
            return None
    
    def _check_connected_devices_macos(self):
        """Check already connected devices on macOS."""
        try:
            devices = self._get_current_devices_macos()
            for device in devices:
                if self._is_target_device_macos(device):
                    logger.info("Found already connected target device")
                    self._handle_device_connected_macos(device)
                    break
        except Exception as e:
            logger.error(f"Error checking connected devices on macOS: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.monitoring:
            self.stop_monitoring()