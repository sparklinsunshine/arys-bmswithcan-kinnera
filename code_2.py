import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import json

# ===============================
# 1. Simulation configuration
# ===============================

DT = 0.1
T_END = 1000.0
N_STEPS = int(T_END / DT) + 1
time = np.linspace(0.0, T_END, N_STEPS)

# ===============================
# 2. Cell & pack parameters
# ===============================

V_MIN_CELL = 3.0
V_MAX_CELL = 4.2
C_CELL_AH = 2.5
R_CELL = 0.05

N_SERIES = 2
N_PARALLEL = 2

C_PACK_AH = C_CELL_AH * N_PARALLEL
R_STRING = R_CELL * N_SERIES
R_PACK = R_STRING / N_PARALLEL
C_PACK_C = C_PACK_AH * 3600.0

# ===============================
# 3. Thermal model parameters
# ===============================

T_AMB = 25.0
T_INIT = 25.0
K_HEAT = 0.001
K_COOL = 0.0005

# Enhanced cooling when active
K_COOL_ACTIVE = 0.002  # Active cooling is 4x more effective

# ===============================
# 4. Protection thresholds
# ===============================

V_OV_CELL = 4.25
V_UV_CELL = 2.90

T_OV = 60.0
T_UV = 0.0

I_OC = 12.0

# Cooling hysteresis
T_COOL_ON = 50.0
T_COOL_OFF = 45.0

# Warning thresholds (before fault)
V_WARN_HIGH_CELL = 4.15
V_WARN_LOW_CELL = 3.10
T_WARN_HIGH = 55.0
I_WARN = 10.0

# ===============================
# 5. VCU State Machine
# ===============================

class VCUState(Enum):
    NORMAL = 0
    WARNING = 1
    COOLING = 2
    SHUTDOWN = 3
    FAULT_LATCHED = 4

@dataclass
class FaultStatus:
    """Comprehensive fault tracking"""
    overV: bool = False
    underV: bool = False
    overT: bool = False
    underT: bool = False
    overI: bool = False
    
    # Warnings (pre-fault conditions)
    warn_highV: bool = False
    warn_lowV: bool = False
    warn_highT: bool = False
    warn_highI: bool = False
    
    def has_critical_fault(self) -> bool:
        return self.overV or self.overT or self.overI
    
    def has_any_fault(self) -> bool:
        return self.overV or self.underV or self.overT or self.underT or self.overI
    
    def has_warning(self) -> bool:
        return (self.warn_highV or self.warn_lowV or 
                self.warn_highT or self.warn_highI)
    
    def to_fault_byte(self) -> int:
        """Pack faults into byte"""
        return (int(self.overV) << 0 |
                int(self.underV) << 1 |
                int(self.overT) << 2 |
                int(self.underT) << 3 |
                int(self.overI) << 4)
    
    def to_warning_byte(self) -> int:
        """Pack warnings into byte"""
        return (int(self.warn_highV) << 0 |
                int(self.warn_lowV) << 1 |
                int(self.warn_highT) << 2 |
                int(self.warn_highI) << 3)

@dataclass
class CANFrame:
    """CAN 2.0B standard frame (8 bytes)"""
    can_id: int          # CAN identifier (e.g., 0x100 for BMS base)
    timestamp: float
    
    # Data fields
    pack_voltage: float  # V
    pack_current: float  # A
    soc: float          # %
    temperature: float  # °C
    fault_flags: int    # byte
    warning_flags: int  # byte
    vcu_state: int      # state enum
    cooling_active: int # 0 or 1
    
    def encode(self) -> bytes:
        """
        Encode to 8-byte CAN payload:
        Byte 0-1: Voltage (uint16, 0.01V resolution, 0-655.35V)
        Byte 2-3: Current (int16, 0.1A resolution, -3276.8 to +3276.7A)
        Byte 4:   SOC (uint8, 0-100%)
        Byte 5:   Temperature (int8, 1°C resolution, -128 to +127°C)
        Byte 6:   Fault flags
        Byte 7:   Warning flags (bits 0-3), VCU state (bits 4-6), cooling (bit 7)
        """
        v_scaled = max(0, min(65535, int(self.pack_voltage * 100)))
        i_scaled = max(-32768, min(32767, int(self.pack_current * 10)))
        soc_scaled = max(0, min(100, int(self.soc)))
        t_scaled = max(-128, min(127, int(self.temperature)))
        
        # Pack byte 7: warnings (4 bits) | state (3 bits) | cooling (1 bit)
        byte7 = ((self.warning_flags & 0x0F) |
                 ((self.vcu_state & 0x07) << 4) |
                 ((self.cooling_active & 0x01) << 7))
        
        return bytes([
            (v_scaled >> 8) & 0xFF,
            v_scaled & 0xFF,
            (i_scaled >> 8) & 0xFF if i_scaled >= 0 else ((i_scaled + 65536) >> 8) & 0xFF,
            i_scaled & 0xFF,
            soc_scaled,
            t_scaled & 0xFF,
            self.fault_flags,
            byte7
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'can_id': f'0x{self.can_id:03X}',
            'timestamp': f'{self.timestamp:.1f}',
            'voltage': f'{self.pack_voltage:.2f}',
            'current': f'{self.pack_current:.2f}',
            'soc': f'{self.soc:.1f}',
            'temperature': f'{self.temperature:.1f}',
            'faults': f'0x{self.fault_flags:02X}',
            'warnings': f'0x{self.warning_flags:02X}',
            'state': VCUState(self.vcu_state).name,
            'cooling': bool(self.cooling_active),
            'raw_bytes': self.encode().hex()
        }

# ===============================
# 6. Helper functions
# ===============================

def current_profile(t: float) -> float:
    """Define commanded pack current"""
    if t < 300.0:
        return 5.0
    elif t < 600.0:
        return 0.0
    elif t < 900.0:
        return -3.0
    elif t < 950.0:
        return 15.0  # Overcurrent spike
    else:
        return 2.0

def voltage_to_soc(v_cell: float) -> float:
    """Estimate SOC from cell voltage"""
    soc = (v_cell - V_MIN_CELL) / (V_MAX_CELL - V_MIN_CELL)
    return max(0.0, min(1.0, soc))

def detect_faults(v_cell: float, i_pack: float, t_pack: float) -> FaultStatus:
    """Comprehensive fault and warning detection"""
    status = FaultStatus()
    
    # Critical faults
    status.overV = v_cell > V_OV_CELL
    status.underV = v_cell < V_UV_CELL
    status.overT = t_pack > T_OV
    status.underT = t_pack < T_UV
    status.overI = abs(i_pack) > I_OC
    
    # Pre-fault warnings
    status.warn_highV = (v_cell > V_WARN_HIGH_CELL) and not status.overV
    status.warn_lowV = (v_cell < V_WARN_LOW_CELL) and not status.underV
    status.warn_highT = (t_pack > T_WARN_HIGH) and not status.overT
    status.warn_highI = (abs(i_pack) > I_WARN) and not status.overI
    
    return status

def vcu_state_machine(current_state: VCUState, faults: FaultStatus, 
                      temp: float, shutdown_latched: bool) -> tuple[VCUState, bool]:
    """
    VCU state machine logic
    Returns: (new_state, shutdown_latched)
    """
    # Once latched in shutdown, stay there until manual reset
    if shutdown_latched:
        return VCUState.FAULT_LATCHED, True
    
    # Critical faults trigger immediate latched shutdown
    if faults.has_critical_fault():
        return VCUState.SHUTDOWN, True
    
    # Non-critical faults
    if faults.has_any_fault():
        return VCUState.SHUTDOWN, False
    
    # Cooling required
    if temp > T_COOL_ON:
        return VCUState.COOLING, False
    
    # Warnings present
    if faults.has_warning():
        return VCUState.WARNING, False
    
    # All clear
    if temp < T_COOL_OFF and current_state == VCUState.COOLING:
        return VCUState.NORMAL, False
    
    if current_state == VCUState.WARNING and not faults.has_warning():
        return VCUState.NORMAL, False
    
    # Maintain current state if in COOLING or WARNING
    if current_state in [VCUState.COOLING, VCUState.WARNING]:
        return current_state, False
    
    return VCUState.NORMAL, False

# ===============================
# 7. State arrays
# ===============================

soc_coulomb = np.zeros(N_STEPS)
soc_voltage = np.zeros(N_STEPS)
soc_blended = np.zeros(N_STEPS)
v_pack = np.zeros(N_STEPS)
t_pack = np.zeros(N_STEPS)
i_cmd = np.zeros(N_STEPS)
i_pack = np.zeros(N_STEPS)

# Fault tracking
fault_overV = np.zeros(N_STEPS, dtype=bool)
fault_underV = np.zeros(N_STEPS, dtype=bool)
fault_overT = np.zeros(N_STEPS, dtype=bool)
fault_underT = np.zeros(N_STEPS, dtype=bool)
fault_overI = np.zeros(N_STEPS, dtype=bool)

warn_highV = np.zeros(N_STEPS, dtype=bool)
warn_lowV = np.zeros(N_STEPS, dtype=bool)
warn_highT = np.zeros(N_STEPS, dtype=bool)
warn_highI = np.zeros(N_STEPS, dtype=bool)

# Control signals
shutdown_cmd = np.zeros(N_STEPS, dtype=bool)
cooling_cmd = np.zeros(N_STEPS, dtype=bool)
vcu_states = np.zeros(N_STEPS, dtype=int)

# CAN logging
can_frames: List[CANFrame] = []
can_log_interval = 1.0  # Log every 1 second

# Initial conditions
soc_coulomb[0] = 1.0
soc_voltage[0] = 1.0
soc_blended[0] = 1.0
t_pack[0] = T_INIT
vcu_states[0] = VCUState.NORMAL.value

# ===============================
# 8. Core simulation loop
# ===============================

ALPHA_BLEND = 0.8  # Weight for coulomb counting
shutdown_latched = False
current_vcu_state = VCUState.NORMAL

for k in range(N_STEPS - 1):
    t = time[k]
    
    # 1) Commanded current from profile
    I_ref = current_profile(t)
    i_cmd[k] = I_ref
    
    # 2) Apply shutdown: contactor opens if shutdown commanded
    if shutdown_cmd[k]:
        I = 0.0
    else:
        I = I_ref
    i_pack[k] = I
    
    # 3) SOC update - Coulomb counting
    delta_Q = I * DT
    soc_coulomb[k+1] = max(0.0, min(1.0, soc_coulomb[k] - delta_Q / C_PACK_C))
    
    # 4) Open-circuit voltage
    voc_cell = V_MIN_CELL + (V_MAX_CELL - V_MIN_CELL) * soc_coulomb[k+1]
    voc_pack = N_SERIES * voc_cell
    
    # 5) Terminal voltage with ohmic drop
    v_pack[k+1] = voc_pack - I * R_PACK
    v_cell_est = v_pack[k+1] / N_SERIES
    
    # 6) Voltage-based SOC estimation
    soc_voltage[k+1] = voltage_to_soc(v_cell_est)
    
    # 7) Blended SOC
    soc_blended[k+1] = (ALPHA_BLEND * soc_coulomb[k+1] + 
                        (1 - ALPHA_BLEND) * soc_voltage[k+1])
    
    # 8) Temperature update with active cooling effect
    k_cool_effective = K_COOL_ACTIVE if cooling_cmd[k] else K_COOL
    q_gen = K_HEAT * (I ** 2) * R_PACK
    q_cool = k_cool_effective * (t_pack[k] - T_AMB)
    t_pack[k+1] = t_pack[k] + (q_gen - q_cool) * DT
    
    # 9) Fault detection
    faults = detect_faults(v_cell_est, I, t_pack[k+1])
    
    fault_overV[k+1] = faults.overV
    fault_underV[k+1] = faults.underV
    fault_overT[k+1] = faults.overT
    fault_underT[k+1] = faults.underT
    fault_overI[k+1] = faults.overI
    
    warn_highV[k+1] = faults.warn_highV
    warn_lowV[k+1] = faults.warn_lowV
    warn_highT[k+1] = faults.warn_highT
    warn_highI[k+1] = faults.warn_highI
    
    # 10) VCU state machine
    current_vcu_state, shutdown_latched = vcu_state_machine(
        current_vcu_state, faults, t_pack[k+1], shutdown_latched
    )
    vcu_states[k+1] = current_vcu_state.value
    
    # 11) Control outputs
    shutdown_cmd[k+1] = (current_vcu_state in [VCUState.SHUTDOWN, VCUState.FAULT_LATCHED])
    
    # Cooling with hysteresis (managed by state machine)
    if current_vcu_state == VCUState.COOLING:
        cooling_cmd[k+1] = True
    elif t_pack[k+1] < T_COOL_OFF:
        cooling_cmd[k+1] = False
    else:
        cooling_cmd[k+1] = cooling_cmd[k]
    
    # 12) Generate CAN frame at specified interval
    if k % int(can_log_interval / DT) == 0:
        can_frame = CANFrame(
            can_id=0x100,  # BMS base CAN ID
            timestamp=t,
            pack_voltage=v_pack[k+1],
            pack_current=I,
            soc=soc_blended[k+1] * 100.0,
            temperature=t_pack[k+1],
            fault_flags=faults.to_fault_byte(),
            warning_flags=faults.to_warning_byte(),
            vcu_state=current_vcu_state.value,
            cooling_active=int(cooling_cmd[k+1])
        )
        can_frames.append(can_frame)

# Final step
i_cmd[-1] = current_profile(time[-1])
i_pack[-1] = 0.0 if shutdown_cmd[-1] else i_cmd[-1]

# ===============================
# 9. Advanced plotting
# ===============================

fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(6, 2, hspace=0.3, wspace=0.3)

# Current profile
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, i_cmd, 'b-', label='I_cmd', linewidth=1.5, alpha=0.7)
ax1.plot(time, i_pack, 'r--', label='I_pack (after shutdown)', linewidth=2)
ax1.axhline(I_OC, color='red', linestyle=':', label='Overcurrent limit')
ax1.axhline(-I_OC, color='red', linestyle=':')
ax1.axhline(I_WARN, color='orange', linestyle=':', label='Warning level')
ax1.axhline(-I_WARN, color='orange', linestyle=':')
ax1.set_ylabel("Current [A]", fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Pack voltage
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time, v_pack, 'g-', linewidth=1.5)
v_ov_pack = V_OV_CELL * N_SERIES
v_uv_pack = V_UV_CELL * N_SERIES
v_warn_high_pack = V_WARN_HIGH_CELL * N_SERIES
v_warn_low_pack = V_WARN_LOW_CELL * N_SERIES
ax2.axhline(v_ov_pack, color='red', linestyle=':', label='Overvoltage')
ax2.axhline(v_uv_pack, color='red', linestyle=':', label='Undervoltage')
ax2.axhline(v_warn_high_pack, color='orange', linestyle=':', alpha=0.7)
ax2.axhline(v_warn_low_pack, color='orange', linestyle=':', alpha=0.7)
ax2.set_ylabel("Pack Voltage [V]", fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# SOC comparison
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(time, soc_coulomb * 100, 'b-', linewidth=1, alpha=0.6, label='Coulomb counting')
ax3.plot(time, soc_voltage * 100, 'r-', linewidth=1, alpha=0.6, label='Voltage-based')
ax3.plot(time, soc_blended * 100, 'g-', linewidth=2, label='Blended estimate')
ax3.set_ylabel("SOC [%]", fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# Temperature
ax4 = fig.add_subplot(gs[3, :])
ax4.plot(time, t_pack, 'm-', linewidth=1.5)
ax4.axhline(T_OV, color='red', linestyle=':', label='Overtemp fault')
ax4.axhline(T_WARN_HIGH, color='orange', linestyle=':', label='Temp warning')
ax4.axhline(T_COOL_ON, color='blue', linestyle=':', label='Cooling on')
ax4.axhline(T_COOL_OFF, color='cyan', linestyle=':', alpha=0.7, label='Cooling off')
ax4.set_ylabel("Temperature [°C]", fontsize=11, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

# Fault flags
ax5 = fig.add_subplot(gs[4, :])
fault_any = fault_overV | fault_underV | fault_overT | fault_underT | fault_overI
warning_any = warn_highV | warn_lowV | warn_highT | warn_highI

ax5.fill_between(time, 0, fault_any.astype(int), alpha=0.3, color='red', label='Fault active')
ax5.fill_between(time, 0, warning_any.astype(int), alpha=0.3, color='orange', label='Warning active')
ax5.plot(time, shutdown_cmd.astype(int), 'r-', linewidth=2, label='Shutdown')
ax5.plot(time, cooling_cmd.astype(int), 'b-', linewidth=2, label='Cooling')
ax5.set_ylabel("Control Flags", fontsize=11, fontweight='bold')
ax5.set_ylim([-0.1, 1.3])
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)

# VCU state
ax6 = fig.add_subplot(gs[5, :])
ax6.plot(time, vcu_states, 'k-', linewidth=2, drawstyle='steps-post')
ax6.set_ylabel("VCU State", fontsize=11, fontweight='bold')
ax6.set_xlabel("Time [s]", fontsize=11, fontweight='bold')
ax6.set_yticks([0, 1, 2, 3, 4])
ax6.set_yticklabels(['NORMAL', 'WARNING', 'COOLING', 'SHUTDOWN', 'FAULT_LATCHED'], fontsize=9)
ax6.grid(True, alpha=0.3)

plt.savefig('bms_simulation_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# 10. Summary & CAN log export
# ===============================

print("\n" + "="*70)
print(" BMS SIMULATION SUMMARY REPORT")
print("="*70)
print(f"Simulation Duration:    {T_END:.1f} seconds")
print(f"Time Step:              {DT:.3f} seconds")
print(f"Pack Configuration:     {N_SERIES}s{N_PARALLEL}p ({C_PACK_AH:.1f} Ah)")
print(f"\n" + "-"*70)
print(" FINAL STATE")
print("-"*70)
print(f"SOC (blended):          {soc_blended[-1]*100:.2f}%")
print(f"Pack Voltage:           {v_pack[-1]:.3f} V")
print(f"Temperature:            {t_pack[-1]:.2f} °C")
print(f"VCU State:              {VCUState(vcu_states[-1]).name}")
print(f"Shutdown Active:        {'YES' if shutdown_cmd[-1] else 'NO'}")
print(f"Cooling Active:         {'YES' if cooling_cmd[-1] else 'NO'}")

print(f"\n" + "-"*70)
print(" OPERATIONAL STATISTICS")
print("-"*70)
print(f"Max Temperature:        {np.max(t_pack):.2f} °C")
print(f"Min Voltage:            {np.min(v_pack):.3f} V")
print(f"Max Voltage:            {np.max(v_pack):.3f} V")
print(f"Max Current (abs):      {np.max(np.abs(i_pack)):.2f} A")
print(f"Energy Throughput:      {np.sum(np.abs(i_pack)) * DT / 3600:.3f} Ah")

print(f"\n" + "-"*70)
print(" FAULT STATISTICS")
print("-"*70)
fault_counts = {
    'Overvoltage': np.sum(fault_overV),
    'Undervoltage': np.sum(fault_underV),
    'Overtemperature': np.sum(fault_overT),
    'Undertemperature': np.sum(fault_underT),
    'Overcurrent': np.sum(fault_overI)
}

for fault_name, count in fault_counts.items():
    if count > 0:
        duration = count * DT
        print(f"{fault_name:20s} {count:5d} steps ({duration:.1f}s)")

print(f"\n" + "-"*70)
print(" WARNING STATISTICS")
print("-"*70)
warning_counts = {
    'High Voltage': np.sum(warn_highV),
    'Low Voltage': np.sum(warn_lowV),
    'High Temperature': np.sum(warn_highT),
    'High Current': np.sum(warn_highI)
}

for warn_name, count in warning_counts.items():
    if count > 0:
        duration = count * DT
        print(f"{warn_name:20s} {count:5d} steps ({duration:.1f}s)")

print(f"\n" + "-"*70)
print(" VCU STATE DISTRIBUTION")
print("-"*70)
for state in VCUState:
    count = np.sum(vcu_states == state.value)
    percentage = (count / N_STEPS) * 100
    duration = count * DT
    if count > 0:
        print(f"{state.name:15s} {count:6d} steps ({duration:7.1f}s, {percentage:5.1f}%)")

print(f"\n" + "-"*70)
print(f" CAN FRAME LOG ({len(can_frames)} frames)")
print("-"*70)
print("\nFirst 3 frames:")
for i, frame in enumerate(can_frames[:3]):
    print(f"\nFrame {i+1} @ t={frame.timestamp:.1f}s:")
    frame_dict = frame.to_dict()
    for key, value in frame_dict.items():
        print(f"  {key:15s} {value}")

print("\nLast frame:")
if can_frames:
    frame = can_frames[-1]
    print(f"\nFrame @ t={frame.timestamp:.1f}s:")
    frame_dict = frame.to_dict()
    for key, value in frame_dict.items():
        print(f"  {key:15s} {value}")

# Export CAN log to JSON
can_log_file = 'bms_can_log.json'
with open(can_log_file, 'w') as f:
    json.dump([frame.to_dict() for frame in can_frames], f, indent=2)
print(f"\n✓ CAN log exported to: {can_log_file}")

# Export summary statistics
stats_file = 'bms_summary.txt'
with open(stats_file, 'w') as f:
    f.write("BMS SIMULATION SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Final SOC: {soc_blended[-1]*100:.2f}%\n")
    f.write(f"Max Temperature: {np.max(t_pack):.2f}°C\n")
    f.write(f"Total Faults: {sum(fault_counts.values())}\n")
    f.write(f"Total Warnings: {sum(warning_counts.values())}\n")
print(f"✓ Summary exported to: {stats_file}")

print("\n" + "="*70 + "\n")