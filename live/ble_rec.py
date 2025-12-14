import asyncio
import struct
import time
from typing import Dict, Optional
import aiohttp

from bleak import BleakClient, BleakScanner

# BLE Service and Characteristic UUIDs (from device.ino)
SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
CHARACTERISTIC_UUID = "12345678-1234-1234-1234-123456789abd"

DEVICE_NAMES = ["XIAO-GYRO-1", "XIAO-GYRO-2"]
DEVICE_ADDRESSES = ["753E1AA1-3AD1-DEF4-5B4A-CF09F9640206", "82DE52DE-64EA-A879-CABF-1B2041FBD922"]

# FastAPI Configuration
API_HOST = "http://127.0.0.1:8087"
API_ENDPOINT = f"{API_HOST}/ble/position"

# Map device names to axis (XIAO-GYRO-1 -> x, XIAO-GYRO-2 -> y)
DEVICE_AXIS_MAP = {
    "XIAO-GYRO-1": "x",
    "XIAO-GYRO-2": "y",
}


class MovementConverter:
    """Convert pitch angle to stable -1 to 1 value."""
    
    def __init__(self, alpha: float = 0.15, max_angle: float = 90.0) -> None:
        """
        Args:
            alpha: Low-pass filter coefficient (0-1). Lower = more smoothing, more stable.
            max_angle: Maximum angle in degrees to map to ±1. Angles beyond this are clamped.
        """
        self.alpha = alpha
        self.max_angle = max_angle
        self.filtered_value = 0.0
        
    def convert(self, pitch: float) -> float:
        """
        Convert pitch angle to -1 to 1 value.
        
        Args:
            pitch: Pitch angle in degrees
            
        Returns:
            Normalized value between -1 and 1
        """
        # Normalize to -1 to 1
        normalized = max(-1.0, min(1.0, pitch / self.max_angle))
        
        # Apply low-pass filter for stability
        self.filtered_value = self.alpha * normalized + (1 - self.alpha) * self.filtered_value
        
        return self.filtered_value


class DeviceConnection:
    def __init__(self, name: str, address: str, session: aiohttp.ClientSession, loop: asyncio.AbstractEventLoop) -> None:
        self.name = name
        self.address = address
        self.client: Optional[BleakClient] = None
        self.connected = False
        self.session = session
        self.loop = loop
        self.axis = DEVICE_AXIS_MAP.get(name, "x")
        self.movement_converter = MovementConverter(alpha=0.15, max_angle=90.0)
        self.last_pitch = 0.0
        self.last_movement = 0.0
        self.last_update_time = 0.0
        self.update_interval = 0.1  # 100ms throttle
        self.pending_value = None
        self._update_task: Optional[asyncio.Task] = None

    def create_imu_handler(self) -> callable:
        """Create a data handler that includes device identification."""
        def handler(sender, data: bytearray) -> None:
            # Schedule the async handler on the event loop
            asyncio.run_coroutine_threadsafe(self.imu_data_handler(data), self.loop)
        return handler

    async def imu_data_handler(self, data: bytearray) -> None:
        # Data format: 24 bytes = 6 * float (yaw, pitch, roll, accelX, accelY, accelZ)
        if len(data) != 24:
            print(f"\n[{self.name}] ⚠️  Unexpected data length: {len(data)} bytes (expected 24)")
            return

        yaw, pitch, roll, accel_x, accel_y, accel_z = struct.unpack("<ffffff", data)
        
        # Convert pitch to -1 to 1 value
        movement_value = self.movement_converter.convert(pitch)
        
        # Store latest values for display
        self.last_pitch = pitch
        self.last_movement = movement_value
        
        # Store pending value to send (throttled)
        self.pending_value = movement_value
        
        # Send immediately if enough time has passed, otherwise it will be sent by the update task
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            await self._send_update()
    
    async def _send_update(self) -> None:
        """Send the pending value to the API"""
        if self.pending_value is None:
            return
        
        try:
            payload = {self.axis: self.pending_value}
            async with self.session.post(API_ENDPOINT, json=payload) as resp:
                if resp.status != 200:
                    print(f"\n[{self.name}] ⚠️  API request failed: {resp.status}")
            self.last_update_time = time.time()
            self.pending_value = None  # Clear after successful send
        except Exception as e:
            print(f"\n[{self.name}] ⚠️  Error sending to API: {e}")
    
    async def _throttled_update_loop(self) -> None:
        """Background task to send throttled updates"""
        while self.connected:
            await asyncio.sleep(self.update_interval)
            if self.pending_value is not None:
                await self._send_update() 

    async def connect(self, device) -> bool:
        """Connect to a discovered device."""
        print(f"Attempting to connect to {self.name} at {device.address}...")
        self.client = BleakClient(device.address)
        try:
            print(f"  [{self.name}] Connecting...")
            await self.client.connect()
            self.connected = True
            print(f"✅ [{self.name}] Connected successfully!")

            print(f"  [{self.name}] Subscribing to characteristic {CHARACTERISTIC_UUID}...")
            await self.client.start_notify(CHARACTERISTIC_UUID, self.create_imu_handler())
            print(f"📡 [{self.name}] Subscribed to IMU data. Streaming to API (throttled to {self.update_interval*1000:.0f}ms)...")
            # Start throttled update task
            self._update_task = asyncio.create_task(self._throttled_update_loop())
            return True
        except Exception as e:
            print(f"❌ [{self.name}] Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        if self.client and self.connected:
            self.connected = False  # Stop the update loop
            # Cancel update task
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass
            try:
                await self.client.stop_notify(CHARACTERISTIC_UUID)
            except Exception:
                pass
            await self.client.disconnect()
            print(f"[{self.name}] Disconnected")


class GyroReceiver:
    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self.devices: Dict[str, DeviceConnection] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._status_task: Optional[asyncio.Task] = None
        
    async def __aenter__(self):
        self.loop = asyncio.get_event_loop()
        self.session = aiohttp.ClientSession()
        for name, address in zip(DEVICE_NAMES, DEVICE_ADDRESSES):
            self.devices[name] = DeviceConnection(name, address, self.session, self.loop)
        print(f"📡 API client configured: {API_ENDPOINT}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def scan_and_connect(self) -> bool:
        print("Scanning for BLE devices...")
        discovered_devices = await BleakScanner.discover(timeout=10.0)

        if not discovered_devices:
            print("No BLE devices found!")
            return False

        print(f"\nFound {len(discovered_devices)} BLE device(s):")
        print("-" * 60)
        for i, d in enumerate(discovered_devices, 1):
            name = d.name if d.name else "Unknown Device"
            print(f"{i:2d}. {name:<25} | {d.address}")
        print("-" * 60)

        # Find target devices by address or name
        target_devices: Dict[str, any] = {}
        
        for device_conn in self.devices.values():
            target_device = None
            
            # Try to match by address first
            for discovered in discovered_devices:
                if discovered.address.lower() == device_conn.address.lower():
                    target_device = discovered
                    print(f"✅ [{device_conn.name}] Found by address: {discovered.name or '(Unknown)'} ({discovered.address})")
                    break
            
            # If not found by address, try by name
            if not target_device:
                for discovered in discovered_devices:
                    if discovered.name == device_conn.name:
                        target_device = discovered
                        print(f"✅ [{device_conn.name}] Found by name: {discovered.name} ({discovered.address})")
                        break
            
            if target_device:
                target_devices[device_conn.name] = target_device
            else:
                print(f"❌ [{device_conn.name}] Could not find device")

        if not target_devices:
            print("\n❌ Could not find any target devices.")
            return False

        # Connect to all devices concurrently
        print("\nConnecting to devices...")
        connection_tasks = [
            self.devices[name].connect(target_devices[name])
            for name in target_devices.keys()
        ]
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        connected_count = sum(1 for r in results if r is True)
        print(f"\n✅ Connected to {connected_count}/{len(target_devices)} device(s)\n")
        
        # Show which devices are connected
        for name, device in self.devices.items():
            status = "✅ Connected" if device.connected else "❌ Not connected"
            print(f"  {status}: {name} ({device.axis.upper()}-axis)")
        
        # Start status display task if we have connected devices
        if connected_count > 0:
            self._status_task = asyncio.create_task(self._status_display_loop())
        
        return connected_count > 0
    
    async def _status_display_loop(self) -> None:
        """Display status of both devices on the same line"""
        while self.has_connected_devices:
            # Build status line with both devices
            status_parts = []
            for name in DEVICE_NAMES:
                if name in self.devices:
                    device = self.devices[name]
                    if device.connected:
                        status_parts.append(
                            f"{name}: Pitch={device.last_pitch:7.2f}° "
                            f"Move={device.last_movement:6.3f} ({device.axis}={device.last_movement:.3f})"
                        )
                    else:
                        status_parts.append(f"{name}: ❌ Disconnected")
            
            if status_parts:
                # Use \r to overwrite the same line
                status_line = " | ".join(status_parts)
                print(f"\r{status_line}", end="", flush=True)
            
            await asyncio.sleep(0.1)  # Update 10 times per second

    async def disconnect_all(self) -> None:
        """Disconnect from all devices."""
        # Stop status display
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
        # Print newline to clear the status line
        print()  # Move to new line after status display stops
        disconnect_tasks = [device.disconnect() for device in self.devices.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    @property
    def has_connected_devices(self) -> bool:
        """Check if any devices are still connected."""
        return any(device.connected for device in self.devices.values())


async def main() -> None:
    async with GyroReceiver() as receiver:
        try:
            if await receiver.scan_and_connect():
                print("\nPress Ctrl+C to stop logging...\n")
                while receiver.has_connected_devices:
                    await asyncio.sleep(1)
            else:
                print("Failed to connect to any devices. Exiting...")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await receiver.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
