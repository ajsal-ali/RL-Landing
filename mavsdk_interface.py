#!/usr/bin/env python3
# MAVSDK interface: provides telemetry, reset and velocity control for the drone via MAVSDK.
"""
MAVSDK interface implementation using user's reference telemetry code.

mavsdk_interface.py: Interface for MAVSDK drone control and telemetry, including reset and velocity commands for RL environment.
"""
import asyncio
import time
import subprocess
import re
import math
from typing import Dict, Optional
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
from mavsdk.offboard import PositionNedYaw


def try_attrs(obj, *names, default=None):
    """Helper function from user's code."""
    if obj is None:
        return default
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return default


def safe_get_float(obj, *names, default=float("nan")):
    """Helper function from user's code."""
    v = try_attrs(obj, *names, default=None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


class MAVSDKInterface:
    """Interface for MAVSDK drone control and telemetry using user's reference implementation."""

    def __init__(self):
        """Initialize MAVSDK interface."""
        self.drone = System()
        self._connected = False
        self._offboard_active = False

        # Telemetry storage (similar to shared_state in user's code)
        self._telemetry_data = {
            'odometry': {},
            'euler': {},
            'velocity_ned': {}
        }

        # Background telemetry tasks
        self._telemetry_tasks = []
        self._stop_telemetry = False

        print("✅ MAVSDKInterface initialized")

    async def connect(self, system_address: str = "udp://:14540") -> None:
        """Connect to PX4 via MAVSDK."""
        if not self._connected:
            await self.drone.connect(system_address=system_address)

            # Wait for connection
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print("✅ MAVSDK connected to PX4")
                    self._connected = True
                    break

            # Start telemetry tasks
            await self._start_telemetry_tasks()

    async def _start_telemetry_tasks(self):
        """Start background telemetry tasks based on user's code."""
        self._stop_telemetry = False
        self._telemetry_tasks = [
            asyncio.create_task(self._telemetry_odometry()),
            asyncio.create_task(self._telemetry_velocity_ned()),
            asyncio.create_task(self._telemetry_euler())
        ]
        print("🔧 Started telemetry background tasks")

    async def _telemetry_odometry(self):
        """Odometry telemetry task from user's code."""
        try:
            if not hasattr(self.drone.telemetry, "odometry"):
                return
            async for odom in self.drone.telemetry.odometry():
                if self._stop_telemetry:
                    break

                # Extract position (px, py, pz) as in user's code
                self._telemetry_data['odometry']['px'] = safe_get_float(try_attrs(odom, 'position_body', 'position'), 'x_m', 'x')
                self._telemetry_data['odometry']['py'] = safe_get_float(try_attrs(odom, 'position_body', 'position'), 'y_m', 'y')
                self._telemetry_data['odometry']['pz'] = safe_get_float(try_attrs(odom, 'position_body', 'position'), 'z_m', 'z')
        except Exception:
            pass

    async def _telemetry_velocity_ned(self):
        """Velocity NED telemetry task from user's code."""
        try:
            if not hasattr(self.drone.telemetry, "velocity_ned"):
                return
            async for v in self.drone.telemetry.velocity_ned():
                if self._stop_telemetry:
                    break

                # Extract NED velocities as in user's code
                vn_num = getattr(v, "north_m_s", getattr(v, "north", float("nan")))
                ve_num = getattr(v, "east_m_s", getattr(v, "east", float("nan")))
                vd_num = getattr(v, "down_m_s", getattr(v, "down", float("nan")))

                # Store raw NED
                self._telemetry_data['velocity_ned']['vn_num'] = float(vn_num) if not math.isnan(vn_num) else float('nan')
                self._telemetry_data['velocity_ned']['ve_num'] = float(ve_num) if not math.isnan(ve_num) else float('nan')
                self._telemetry_data['velocity_ned']['vd_num'] = float(vd_num) if not math.isnan(vd_num) else float('nan')

                # Compute body-frame vxp, vyp from NED velocities using current yaw (exactly as in user's code)
                yaw_val = self._telemetry_data.get('euler', {}).get('yaw_deg', None)
                try:
                    yaw_deg = float(yaw_val) if yaw_val is not None else float('nan')
                except Exception:
                    yaw_deg = float('nan')

                if not (math.isnan(vn_num) or math.isnan(ve_num) or math.isnan(yaw_deg)):
                    yaw_rad = math.radians(yaw_deg)
                    vxp = vn_num * math.cos(yaw_rad) + ve_num * math.sin(yaw_rad)   # forward (body x)
                    vyp = -vn_num * math.sin(yaw_rad) + ve_num * math.cos(yaw_rad)  # right (body y)
                else:
                    vxp = float('nan')
                    vyp = float('nan')

                # Store body-frame velocities (vxp, vyp as requested)
                self._telemetry_data['velocity_ned']['vxp_num'] = float(vxp) if not math.isnan(vxp) else float('nan')
                self._telemetry_data['velocity_ned']['vyp_num'] = float(vyp) if not math.isnan(vyp) else float('nan')

        except Exception:
            pass

    async def _telemetry_euler(self):
        """Euler attitude telemetry task from user's code."""
        try:
            gen = None
            if hasattr(self.drone.telemetry, "attitude_euler"):
                gen = self.drone.telemetry.attitude_euler()
            elif hasattr(self.drone.telemetry, "euler_angle"):
                gen = self.drone.telemetry.euler_angle()
            if gen is None:
                return

            async for e in gen:
                if self._stop_telemetry:
                    break

                # Extract roll, pitch, yaw as in user's code
                self._telemetry_data['euler']['roll_deg'] = safe_get_float(e, 'roll_deg', 'roll')
                self._telemetry_data['euler']['pitch_deg'] = safe_get_float(e, 'pitch_deg', 'pitch')
                self._telemetry_data['euler']['yaw_deg'] = safe_get_float(e, 'yaw_deg', 'yaw')
        except Exception:
            pass

    async def reset_to_position(self, x: float, y: float, z: float) -> None:
        """Reset drone to specified position and arm/takeoff."""
        await self.connect()

        print(f"🔄 Resetting drone to position ({x:.2f}, {y:.2f}, {z:.2f})")

        try:
            # Stop offboard mode if active
            if self._offboard_active:
                await self.drone.offboard.stop()
                self._offboard_active = False
                await asyncio.sleep(1.0)

            print("🔧 Arming and taking off...")
            # Arm if not armed
            try:
                await self.drone.action.arm()
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"Arm failed (may already be armed): {e}")

            # Start offboard mode for position control
            print("🎯 Starting offboard mode and navigating to target...")
            await self._start_offboard_mode()

            # Use position setpoint to go to target
            await self.drone.offboard.set_position_ned(PositionNedYaw(x, y, -z, 0.0))  # -z for altitude
            await asyncio.sleep(8.0)  # Give time for navigation

            print("🚀 Navigation complete")
            telemetry = await self.get_telemetry()
            drone_x = telemetry.get('x', 0.0)
            drone_y = telemetry.get('y', 0.0)
            print(f"Current position: x={drone_x:.2f}, y={drone_y:.2f}")

            # Stop at target position with zero velocity
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
            await asyncio.sleep(1.0)

        except Exception as e:
            print(f"❌ Reset failed: {e}")
            # Continue anyway for training

    async def set_velocity(self, vx: float, vy: float, vz: float) -> None:
        """Set drone velocity setpoint."""
        if not self._offboard_active:
            await self._start_offboard_mode()

        # Clamp velocities for safety
        vx = max(-2.0, min(2.0, vx))
        vy = max(-2.0, min(2.0, vy))
        vz = max(-1.0, min(1.0, vz))

        try:
            velocity_cmd = VelocityBodyYawspeed(vx, vy, vz, 0.0)  # 0 yaw rate
            await self.drone.offboard.set_velocity_body(velocity_cmd)
        except Exception as e:
            print(f"⚠️ Velocity command failed: {e}")

    async def get_telemetry(self) -> Dict[str, float]:
        """Get current telemetry data using user's reference implementation."""
        telemetry_data = {
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'altitude': 2.0,
            'pitch': 0.0, 'roll': 0.0, 'vx': 0.0, 'vy': 0.0
        }

        try:
            # Use odometry position (px, py, pz) as in user's code
            odom = self._telemetry_data.get('odometry', {})
            px = odom.get('px', 0.0)
            py = odom.get('py', 0.0)
            pz = odom.get('pz', 2.0)

            if not math.isnan(px):
                telemetry_data['x'] = px
            if not math.isnan(py):
                telemetry_data['y'] = py
            if not math.isnan(pz):
                telemetry_data['z'] = pz
                telemetry_data['altitude'] = max(0.1, abs(pz))  # Altitude above ground

            # Use euler angles (roll, pitch) as in user's code
            euler = self._telemetry_data.get('euler', {})
            roll_deg = euler.get('roll_deg', 0.0)
            pitch_deg = euler.get('pitch_deg', 0.0)

            if not math.isnan(roll_deg):
                telemetry_data['roll'] = math.radians(roll_deg)  # Convert to radians
            if not math.isnan(pitch_deg):
                telemetry_data['pitch'] = math.radians(pitch_deg)  # Convert to radians

            # Use body-frame velocities (vxp, vyp) as requested by user
            vel_ned = self._telemetry_data.get('velocity_ned', {})
            vxp = vel_ned.get('vxp_num', 0.0)
            vyp = vel_ned.get('vyp_num', 0.0)

            if not math.isnan(vxp):
                telemetry_data['vx'] = vxp
            if not math.isnan(vyp):
                telemetry_data['vy'] = vyp

        except Exception as e:
            print(f"⚠️ Telemetry read failed: {e}")

        return telemetry_data

    async def _get_gazebo_position(self) -> Dict[str, float]:
        """Get position from Gazebo topic (fallback method)."""
        try:
            # Run gz topic command to get position
            cmd = ["gz", "topic", "-e", "-t", "/world/onlypad/pose/info", "-n", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)

            if result.returncode == 0:
                # Parse output for x500_0 position
                if '"x500_0"' in result.stdout:
                    # Extract position block
                    match = re.search(r"position\s*\{([^}]*)\}", result.stdout, re.DOTALL)
                    if match:
                        pos_text = match.group(1)
                        coords = {}
                        for axis in ['x', 'y', 'z']:
                            axis_match = re.search(f"{axis}\s*:\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", pos_text)
                            if axis_match:
                                coords[axis] = float(axis_match.group(1))
                        if coords:
                            return coords
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass

        # Return default if parsing fails
        return {'x': 0.0, 'y': 0.0, 'z': 2.0}

    async def _start_offboard_mode(self) -> None:
        """Start offboard mode for velocity control (based on user's prestream method)."""
        if self._offboard_active:
            return

        try:
            # Pre-stream velocity commands before starting offboard (user's method)
            print("🔧 Pre-streaming velocity commands...")
            duration_s = 1.0
            rate_hz = 20.0
            period = 1.0 / rate_hz
            t_end = time.monotonic() + duration_s

            while time.monotonic() < t_end:
                await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
                await asyncio.sleep(period)

            # Start offboard mode
            await self.drone.offboard.start()
            self._offboard_active = True
            print("✅ Offboard mode started")

        except OffboardError as e:
            print(f"❌ Offboard start failed: {e}")
            raise RuntimeError(f"Offboard start failed: {e}")

    async def cleanup(self):
        """Cleanup telemetry tasks."""
        self._stop_telemetry = True
        for task in self._telemetry_tasks:
            task.cancel()
        await asyncio.gather(*self._telemetry_tasks, return_exceptions=True)
