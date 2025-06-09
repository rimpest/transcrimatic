"""
Unit tests for the DeviceMonitor module.
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Mock pyudev before importing device_monitor
sys.modules['pyudev'] = MagicMock()

from src.device.device_monitor import DeviceMonitor, DeviceEvent, DeviceInfo
from src.config import ConfigManager


class TestDeviceMonitor:
    """Test suite for DeviceMonitor class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ConfigManager."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            "device.identifier": "Sony_ICD-TX660",
            "device.vendor_id": "054c",
            "device.product_id": "0xxx",
            "device.serial_pattern": "^[A-Z0-9]{8}$"
        }.get(key, default)
        return config
    
    @pytest.fixture
    def device_monitor(self, mock_config):
        """Create a DeviceMonitor instance."""
        return DeviceMonitor(mock_config)
    
    def test_init(self, device_monitor, mock_config):
        """Test DeviceMonitor initialization."""
        assert device_monitor.config == mock_config
        assert device_monitor.callback is None
        assert device_monitor.monitoring is False
        assert device_monitor.monitor_thread is None
        assert device_monitor.target_identifier == "Sony_ICD-TX660"
        assert device_monitor.target_vendor_id == "054c"
        assert device_monitor.target_product_id == "0xxx"
        assert device_monitor.serial_pattern == "^[A-Z0-9]{8}$"
    
    def test_start_monitoring(self, device_monitor):
        """Test starting device monitoring."""
        callback = Mock()
        
        with patch.object(device_monitor, '_check_connected_devices'):
            device_monitor.start_monitoring(callback)
        
        assert device_monitor.monitoring is True
        assert device_monitor.callback == callback
        assert device_monitor.monitor_thread is not None
        assert device_monitor.monitor_thread.is_alive()
        
        # Clean up
        device_monitor.stop_monitoring()
    
    def test_start_monitoring_already_active(self, device_monitor, caplog):
        """Test starting monitoring when already active."""
        device_monitor.monitoring = True
        callback = Mock()
        
        device_monitor.start_monitoring(callback)
        
        assert "Device monitoring already active" in caplog.text
    
    def test_stop_monitoring(self, device_monitor):
        """Test stopping device monitoring."""
        callback = Mock()
        
        with patch.object(device_monitor, '_check_connected_devices'):
            device_monitor.start_monitoring(callback)
        
        device_monitor.stop_monitoring()
        
        assert device_monitor.monitoring is False
        assert device_monitor._stop_event.is_set()
        assert device_monitor.monitor_thread is None
        assert device_monitor.callback is None
    
    def test_stop_monitoring_not_active(self, device_monitor):
        """Test stopping monitoring when not active."""
        device_monitor.stop_monitoring()  # Should not raise any errors
    
    def test_is_monitoring_property(self, device_monitor):
        """Test is_monitoring property."""
        assert device_monitor.is_monitoring is False
        
        device_monitor.monitoring = True
        assert device_monitor.is_monitoring is True
    
    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_is_target_device(self, device_monitor, platform):
        """Test device matching logic."""
        with patch('sys.platform', platform):
            if platform.startswith('linux'):
                # Mock Linux device
                device = Mock()
                device.get.side_effect = lambda key, default='': {
                    'ID_VENDOR_ID': '054c',
                    'ID_MODEL_ID': '0xxx',
                    'ID_SERIAL_SHORT': 'ABCD1234',
                    'ID_MODEL': 'Sony_ICD-TX660'
                }.get(key, default)
                
                assert device_monitor.is_target_device(device) is True
                
                # Test non-matching device
                device.get.side_effect = lambda key, default='': {
                    'ID_VENDOR_ID': '1234',
                    'ID_MODEL': 'Other_Device'
                }.get(key, default)
                
                assert device_monitor.is_target_device(device) is False
                
            elif platform == 'darwin':
                # Mock macOS diskutil output
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value.stdout = "Device Name: Sony_ICD-TX660\nVendor: Sony"
                    mock_run.return_value.returncode = 0
                    
                    assert device_monitor.is_target_device("/dev/disk2") is True
                    
                    mock_run.return_value.stdout = "Device Name: Other Device"
                    assert device_monitor.is_target_device("/dev/disk2") is False
    
    def test_event_debouncing(self, device_monitor):
        """Test event debouncing logic."""
        # Test that duplicate events are debounced
        assert device_monitor._should_emit_event("/dev/sda", "connected") is True
        assert device_monitor._should_emit_event("/dev/sda", "connected") is False
        
        # Test different event types
        assert device_monitor._should_emit_event("/dev/sda", "disconnected") is True
        
        # Test different devices
        assert device_monitor._should_emit_event("/dev/sdb", "connected") is True
        
        # Test after debounce window
        with patch('time.time', side_effect=[0, 0, 10]):  # Simulate 10 seconds passing
            device_monitor._recent_events.clear()
            assert device_monitor._should_emit_event("/dev/sda", "connected") is True
            assert device_monitor._should_emit_event("/dev/sda", "connected") is False
            assert device_monitor._should_emit_event("/dev/sda", "connected") is True
    
    def test_emit_event(self, device_monitor):
        """Test event emission."""
        callback = Mock()
        device_monitor.callback = callback
        
        event = DeviceEvent(
            event_type="connected",
            device_path="/dev/sda",
            mount_point="/media/recorder",
            timestamp=datetime.now(),
            device_info=None
        )
        
        device_monitor._emit_event(event)
        
        callback.assert_called_once_with(event)
    
    def test_emit_event_no_callback(self, device_monitor, caplog):
        """Test event emission without callback."""
        event = DeviceEvent(
            event_type="connected",
            device_path="/dev/sda",
            mount_point="/media/recorder",
            timestamp=datetime.now(),
            device_info=None
        )
        
        device_monitor._emit_event(event)
        
        assert "No callback registered" in caplog.text
    
    def test_emit_event_callback_error(self, device_monitor, caplog):
        """Test event emission with callback error."""
        callback = Mock(side_effect=Exception("Callback error"))
        device_monitor.callback = callback
        
        event = DeviceEvent(
            event_type="connected",
            device_path="/dev/sda",
            mount_point="/media/recorder",
            timestamp=datetime.now(),
            device_info=None
        )
        
        device_monitor._emit_event(event)
        
        assert "Error in device event callback" in caplog.text
    
    @patch('time.sleep')
    def test_wait_for_mount(self, mock_sleep, device_monitor):
        """Test waiting for device mount."""
        # Test successful mount
        with patch.object(device_monitor, '_get_mount_point', side_effect=[None, None, "/media/recorder"]):
            mount_point = device_monitor._wait_for_mount("/dev/sda", timeout=2)
            assert mount_point == "/media/recorder"
            assert mock_sleep.call_count == 2
        
        # Test timeout
        mock_sleep.reset_mock()
        with patch.object(device_monitor, '_get_mount_point', return_value=None):
            mount_point = device_monitor._wait_for_mount("/dev/sda", timeout=1)
            assert mount_point is None
    
    def test_wait_for_mount_stop_event(self, device_monitor):
        """Test wait for mount with stop event."""
        device_monitor._stop_event.set()
        
        with patch.object(device_monitor, '_get_mount_point', return_value=None):
            mount_point = device_monitor._wait_for_mount("/dev/sda")
            assert mount_point is None
    
    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Linux-specific test")
    def test_get_mount_point_linux(self, device_monitor):
        """Test Linux mount point detection."""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = [
                "/dev/sda1 /boot ext4 rw 0 0\n",
                "/dev/sdb1 /media/user/Sony_ICD-TX660 vfat rw 0 0\n"
            ]
            
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=True):
                    mount_point = device_monitor._get_mount_point_linux("/dev/sdb")
                    assert mount_point == "/media/user/Sony_ICD-TX660"
    
    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_get_mount_point_macos(self, device_monitor):
        """Test macOS mount point detection."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = """
            Device Identifier: disk2
            Mount Point: /Volumes/Sony_ICD-TX660
            Volume Name: Sony_ICD-TX660
            """
            mock_run.return_value.returncode = 0
            
            mount_point = device_monitor._get_mount_point_macos("/dev/disk2")
            assert mount_point == "/Volumes/Sony_ICD-TX660"
    
    def test_get_device_info(self, device_monitor):
        """Test getting device information."""
        with patch.object(device_monitor, '_get_mount_point', return_value="/media/recorder"):
            with patch('shutil.disk_usage') as mock_disk_usage:
                mock_disk_usage.return_value = Mock(total=1000000, free=500000)
                
                if sys.platform.startswith('linux'):
                    with patch.object(device_monitor, '_get_device_info_linux') as mock_get_info:
                        mock_get_info.return_value = DeviceInfo(
                            vendor_id="054c",
                            product_id="0xxx",
                            serial_number="ABCD1234",
                            device_name="Sony_ICD-TX660",
                            mount_point="/media/recorder",
                            total_space=1000000,
                            free_space=500000
                        )
                        
                        info = device_monitor.get_device_info("/dev/sda")
                        assert info is not None
                        assert info.vendor_id == "054c"
                        assert info.mount_point == "/media/recorder"
    
    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Linux-specific test")
    def test_monitor_linux(self, device_monitor):
        """Test Linux USB monitoring loop."""
        # Mock pyudev
        mock_context = Mock()
        mock_monitor = Mock()
        mock_device = Mock()
        
        mock_device.action = 'add'
        mock_device.device_path = '/dev/sda'
        
        mock_monitor.poll.return_value = mock_device
        mock_context.list_devices.return_value = []
        
        with patch('pyudev.Context', return_value=mock_context):
            with patch('pyudev.Monitor.from_netlink', return_value=mock_monitor):
                with patch.object(device_monitor, 'is_target_device', return_value=True):
                    with patch.object(device_monitor, '_handle_device_connected_linux'):
                        # Run monitor for a short time
                        monitor_thread = threading.Thread(target=device_monitor._monitor_linux)
                        monitor_thread.start()
                        
                        time.sleep(0.1)
                        device_monitor.monitoring = False
                        monitor_thread.join(timeout=1)
    
    def test_check_connected_devices(self, device_monitor):
        """Test checking for already connected devices."""
        if sys.platform.startswith('linux'):
            with patch.object(device_monitor, '_check_connected_devices_linux') as mock_check:
                device_monitor._check_connected_devices()
                mock_check.assert_called_once()
        elif sys.platform == 'darwin':
            with patch.object(device_monitor, '_check_connected_devices_macos') as mock_check:
                device_monitor._check_connected_devices()
                mock_check.assert_called_once()
    
    def test_platform_not_supported(self, mock_config):
        """Test unsupported platform handling."""
        with patch('sys.platform', 'win32'):
            monitor = DeviceMonitor(mock_config)
            
            with pytest.raises(NotImplementedError):
                monitor.start_monitoring(Mock())
    
    def test_serial_pattern_matching(self, device_monitor):
        """Test serial number pattern matching."""
        with patch('src.device.device_monitor.sys.platform', 'linux'):
            # Valid serial
            device = Mock()
            device.get.side_effect = lambda key, default='': {
                'ID_VENDOR_ID': '054c',
                'ID_MODEL_ID': '0xxx',
                'ID_MODEL': 'Sony_ICD-TX660',
                'ID_SERIAL_SHORT': 'ABCD1234'
            }.get(key, default)
            
            assert device_monitor.is_target_device(device) is True
            
            # Invalid serial
            device.get.side_effect = lambda key, default='': {
                'ID_VENDOR_ID': '054c',
                'ID_MODEL_ID': '0xxx',
                'ID_MODEL': 'Sony_ICD-TX660',
                'ID_SERIAL_SHORT': 'invalid-serial-123'
            }.get(key, default)
            
            assert device_monitor.is_target_device(device) is False
    
    def test_device_info_creation(self, device_monitor):
        """Test DeviceInfo object creation."""
        info = DeviceInfo(
            vendor_id="054c",
            product_id="0xxx",
            serial_number="ABCD1234",
            device_name="Sony_ICD-TX660",
            mount_point="/media/recorder",
            total_space=1000000000,
            free_space=500000000
        )
        
        assert info.vendor_id == "054c"
        assert info.product_id == "0xxx"
        assert info.serial_number == "ABCD1234"
        assert info.device_name == "Sony_ICD-TX660"
        assert info.mount_point == "/media/recorder"
        assert info.total_space == 1000000000
        assert info.free_space == 500000000
    
    def test_device_event_creation(self):
        """Test DeviceEvent object creation."""
        timestamp = datetime.now()
        info = DeviceInfo(
            vendor_id="054c",
            product_id="0xxx",
            serial_number="ABCD1234",
            device_name="Sony_ICD-TX660",
            mount_point="/media/recorder",
            total_space=1000000000,
            free_space=500000000
        )
        
        event = DeviceEvent(
            event_type="connected",
            device_path="/dev/sda",
            mount_point="/media/recorder",
            timestamp=timestamp,
            device_info=info
        )
        
        assert event.event_type == "connected"
        assert event.device_path == "/dev/sda"
        assert event.mount_point == "/media/recorder"
        assert event.timestamp == timestamp
        assert event.device_info == info