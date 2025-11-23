import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple
import json

# ============================================================
# 1. Simulation configuration
# ============================================================

DT = 0.1
T_END = 1000.0
N_STEPS = int(T_END / DT) + 1

time = np.linspace(0.0, T_END, N_STEPS)

# CAN configuration
CAN_PERIOD = 1.0
BMS_STATUS_ID = 0x100
BMS_DETAIL_ID = 0x101
VCU_CMD_ID = 0x200

# ============================================================
# 2. Cell & pack parameters (2s2p Li-ion pack)
# ============================================================

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

# ============================================================
# 3. Thermal model parameters
# ============================================================

T_AMB = 25.0
T_INIT = 25.0
K_HEAT = 0.001
K_COOL = 0.0005
K_COOL_ACTIVE = 0.002  # Enhanced cooling when active

# ============================================================
# 4. Protection thresholds
# ============================================================

V_OV_CELL = 4.25
V_UV_CELL = 2.90

T_OV = 60.0
T_UV = 0.0

I_OC = 12.0

# Cooling hysteresis
T_COOL_ON = 50.0
T_COOL_OFF = 45.0

# Warning thresholds
V_WARN_HIGH = 4.15
V_WARN_LOW = 3.10
T_WARN = 55.0
I_WARN = 10.0

# ============================================================
# 5. Enums & Data Classes
# ============================================================

class VCUState(Enum):
    NORMAL = 0
    WARNING = 1
    COOLING = 2
    SHUTDOWN = 3
    FAULT_LATCHED = 4

class FaultBit(Enum):
    OVERVOLTAGE = 0
    UNDERVOLTAGE = 1
    OVERTEMP = 2
    UNDERTEMP = 3
    OVERCURRENT = 4
    CELL_IMBALANCE = 5
    RESERVED_6 = 6
    RESERVED_7 = 7

@dataclass
class CANMessage:
    """Standard CAN 2.0B message"""
    timestamp: float
    can_id: int
    data: List[int]  # 8 bytes
    dlc: int = 8     # Data Length Code
    
    def to_hex_string(self) -> str:
        return ' '.join([f'{b:02X}' for b in self.data])
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': f'{self.timestamp:.3f}',
            'can_id': f'0x{self.can_id:03X}',
            'dlc': self.dlc,
            'data': self.to_hex_string(),
            'data_decimal': self.data
        }

# ============================================================
# 6. Helper: current profile
# ============================================================

def current_profile(t: float) -> float:
    """Commanded pack current profile"""
    if t < 300.0:
        return 5.0      # Discharge at 5 A
    elif t < 600.0:
        return 0.0      # Rest
    elif t < 900.0:
        return -3.0     # Charge at -3 A
    elif t < 950.0:
        return 15.0     # Overcurrent spike
    else:
        return 2.0      # Small discharge

# ============================================================
# 7. CAN encoding/decoding functions
# ============================================================

def encode_signed_16(value: float, scale: float = 10.0) -> Tuple[int, int]:
    """
    Encode signed value to 16-bit two's complement bytes
    Args:
        value: floating point value
        scale: scaling factor (e.g., 10 for 0.1 resolution)
    Returns:
        (low_byte, high_byte)
    """
    scaled = int(round(value * scale))
    if scaled < 0:
        scaled = (1 << 16) + scaled
    scaled &= 0xFFFF
    lo = scaled & 0xFF
    hi = (scaled >> 8) & 0xFF
    return lo, hi

def decode_signed_16(lo: int, hi: int, scale: float = 10.0) -> float:
    """Decode 16-bit two's complement bytes to signed value"""
    raw = lo | (hi << 8)
    if raw & 0x8000:  # Check sign bit
        raw -= 1 << 16
    return raw / scale

def encode_unsigned_16(value: float, scale: float = 10.0) -> Tuple[int, int]:
    """Encode unsigned value to 16-bit bytes"""
    scaled = int(round(value * scale))
    scaled = max(0, min(65535, scaled))
    lo = scaled & 0xFF
    hi = (scaled >> 8) & 0xFF
    return lo, hi

def decode_unsigned_16(lo: int, hi: int, scale: float = 10.0) -> float:
    """Decode 16-bit unsigned bytes"""
    raw = lo | (hi << 8)
    return raw / scale

def encode_bms_status_frame(v_pack: float, i_pack: float, soc: float, 
                           faults: int, temp: float) -> List[int]:
    """
    BMS Status Frame (CAN ID 0x100):
    Byte 0-1: Pack voltage [0.1V resolution, unsigned]
    Byte 2-3: Pack current [0.1A resolution, signed]
    Byte 4:   SOC [1% resolution, 0-100]
    Byte 5:   Fault bitfield
    Byte 6:   Temperature [1°C resolution, signed -40 to +85°C offset]
    Byte 7:   Reserved / checksum
    """
    v_lo, v_hi = encode_unsigned_16(v_pack, 10.0)
    i_lo, i_hi = encode_signed_16(i_pack, 10.0)
    
    soc_byte = int(np.clip(round(soc * 100), 0, 100))
    
    # Temperature with offset: -40°C to +85°C -> 0 to 125
    temp_offset = int(np.clip(round(temp + 40), 0, 255))
    
    # Simple checksum (XOR of first 7 bytes)
    checksum = v_lo ^ v_hi ^ i_lo ^ i_hi ^ soc_byte ^ faults ^ temp_offset
    
    return [v_lo, v_hi, i_lo, i_hi, soc_byte, faults, temp_offset, checksum]

def decode_bms_status_frame(data: List[int]) -> Dict:
    """Decode BMS status frame to physical values"""
    v_pack = decode_unsigned_16(data[0], data[1], 10.0)
    i_pack = decode_signed_16(data[2], data[3], 10.0)
    soc_pct = data[4]
    fault_bits = data[5]
    temp_c = data[6] - 40  # Remove offset
    checksum_rx = data[7]
    
    # Verify checksum
    checksum_calc = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5] ^ data[6]
    checksum_valid = (checksum_calc == checksum_rx)
    
    return {
        'voltage': v_pack,
        'current': i_pack,
        'soc': soc_pct,
        'faults': fault_bits,
        'temperature': temp_c,
        'checksum_valid': checksum_valid
    }

def encode_bms_detail_frame(v_cell_min: float, v_cell_max: float, 
                           soh: float, cycles: int) -> List[int]:
    """
    BMS Detail Frame (CAN ID 0x101):
    Byte 0-1: Min cell voltage [1mV resolution]
    Byte 2-3: Max cell voltage [1mV resolution]
    Byte 4:   SOH [1% resolution, 0-100]
    Byte 5-6: Cycle count [0-65535]
    Byte 7:   Reserved
    """
    vmin_lo, vmin_hi = encode_unsigned_16(v_cell_min, 1000.0)
    vmax_lo, vmax_hi = encode_unsigned_16(v_cell_max, 1000.0)
    
    soh_byte = int(np.clip(round(soh * 100), 0, 100))
    
    cycles = max(0, min(65535, cycles))
    cycles_lo = cycles & 0xFF
    cycles_hi = (cycles >> 8) & 0xFF
    
    return [vmin_lo, vmin_hi, vmax_lo, vmax_hi, soh_byte, cycles_lo, cycles_hi, 0x00]

def vcu_state_from_frame(data: List[int], cooling_active: bool) -> VCUState:
    """
    VCU decision logic based on received CAN frame
    Args:
        data: BMS status frame data
        cooling_active: current cooling state
    Returns:
        VCUState enum
    """
    decoded = decode_bms_status_frame(data)
    
    if not decoded['checksum_valid']:
        return VCUState.FAULT_LATCHED
    
    fault_bits = decoded['faults']
    temp = decoded['temperature']
    
    # Critical faults -> shutdown
    critical_faults = (
        (fault_bits & (1 << FaultBit.OVERVOLTAGE.value)) or
        (fault_bits & (1 << FaultBit.OVERTEMP.value)) or
        (fault_bits & (1 << FaultBit.OVERCURRENT.value))
    )
    
    if critical_faults:
        return VCUState.SHUTDOWN
    
    # Any fault present
    if fault_bits != 0:
        return VCUState.WARNING
    
    # Temperature-based cooling
    if temp > T_COOL_ON:
        return VCUState.COOLING
    elif temp < T_COOL_OFF and cooling_active:
        return VCUState.COOLING  # Maintain cooling until temp drops
    elif cooling_active and temp < T_COOL_OFF:
        return VCUState.NORMAL
    
    return VCUState.NORMAL

# ============================================================
# 8. State arrays
# ============================================================

soc_coulomb = np.zeros(N_STEPS)
soc_voltage = np.zeros(N_STEPS)
soc_blended = np.zeros(N_STEPS)
v_pack_arr = np.zeros(N_STEPS)
t_pack = np.zeros(N_STEPS)
i_cmd = np.zeros(N_STEPS)
i_pack = np.zeros(N_STEPS)

# Cell voltages (for imbalance detection)
v_cell_min = np.zeros(N_STEPS)
v_cell_max = np.zeros(N_STEPS)

# Fault flags
fault_overV = np.zeros(N_STEPS, dtype=bool)
fault_underV = np.zeros(N_STEPS, dtype=bool)
fault_overT = np.zeros(N_STEPS, dtype=bool)
fault_underT = np.zeros(N_STEPS, dtype=bool)
fault_overI = np.zeros(N_STEPS, dtype=bool)

# Control signals
shutdown_cmd = np.zeros(N_STEPS, dtype=bool)
cooling_cmd = np.zeros(N_STEPS, dtype=bool)
vcu_states = np.zeros(N_STEPS, dtype=int)

# CAN message log
can_log: List[CANMessage] = []

# State tracking
soc_coulomb[0] = 1.0
soc_voltage[0] = 1.0
soc_blended[0] = 1.0
t_pack[0] = T_INIT
vcu_states[0] = VCUState.NORMAL.value
last_can_time = -CAN_PERIOD
current_vcu_state = VCUState.NORMAL
shutdown_latched = False
cycle_count = 0

# ============================================================
# 9. Core simulation loop
# ============================================================

ALPHA_BLEND = 0.8  # SOC blending weight

for k in range(N_STEPS - 1):
    t = time[k]
    
    # 1) Commanded current from profile
    I_ref = current_profile(t)
    i_cmd[k] = I_ref
    
    # 2) Apply shutdown: if shutdown commanded, force current to zero
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
    v_pack = voc_pack - I * R_PACK
    v_pack_arr[k+1] = v_pack
    v_cell_est = v_pack / N_SERIES
    
    # 6) Voltage-based SOC
    soc_voltage[k+1] = (v_cell_est - V_MIN_CELL) / (V_MAX_CELL - V_MIN_CELL)
    soc_voltage[k+1] = max(0.0, min(1.0, soc_voltage[k+1]))
    
    # 7) Blended SOC
    soc_blended[k+1] = (ALPHA_BLEND * soc_coulomb[k+1] + 
                        (1 - ALPHA_BLEND) * soc_voltage[k+1])
    
    # 8) Cell voltage simulation (with small imbalance)
    imbalance = 0.03 * np.sin(t / 150.0)  # ±30mV variation
    v_cell_min[k+1] = v_cell_est * (1 - imbalance)
    v_cell_max[k+1] = v_cell_est * (1 + imbalance)
    
    # 9) Temperature update with cooling effect
    k_cool_eff = K_COOL_ACTIVE if cooling_cmd[k] else K_COOL
    q_gen = K_HEAT * (I ** 2) * R_PACK
    q_cool = k_cool_eff * (t_pack[k] - T_AMB)
    t_pack[k+1] = t_pack[k] + (q_gen - q_cool) * DT
    
    # 10) Fault detection
    overV = v_cell_est > V_OV_CELL
    underV = v_cell_est < V_UV_CELL
    overT = t_pack[k+1] > T_OV
    underT = t_pack[k+1] < T_UV
    overI = abs(I) > I_OC
    
    fault_overV[k+1] = overV
    fault_underV[k+1] = underV
    fault_overT[k+1] = overT
    fault_underT[k+1] = underT
    fault_overI[k+1] = overI
    
    # 11) Build fault bitfield
    fault_bits = (
        (1 if overV else 0) << FaultBit.OVERVOLTAGE.value |
        (1 if underV else 0) << FaultBit.UNDERVOLTAGE.value |
        (1 if overT else 0) << FaultBit.OVERTEMP.value |
        (1 if underT else 0) << FaultBit.UNDERTEMP.value |
        (1 if overI else 0) << FaultBit.OVERCURRENT.value
    )
    
    # 12) Shutdown logic (critical faults cause latched shutdown)
    if overV or overT or overI:
        shutdown_cmd[k+1] = True
        shutdown_latched = True
    elif shutdown_latched:
        shutdown_cmd[k+1] = True
    else:
        shutdown_cmd[k+1] = False
    
    # 13) Cooling logic with hysteresis
    if t_pack[k+1] > T_COOL_ON:
        cooling_cmd[k+1] = True
    elif t_pack[k+1] < T_COOL_OFF:
        cooling_cmd[k+1] = False
    else:
        cooling_cmd[k+1] = cooling_cmd[k]
    
    # 14) CAN transmission at fixed period
    if t - last_can_time >= CAN_PERIOD:
        last_can_time = t
        
        # Encode BMS status frame
        status_data = encode_bms_status_frame(
            v_pack=v_pack,
            i_pack=I,
            soc=soc_blended[k+1],
            faults=fault_bits,
            temp=t_pack[k+1]
        )
        
        status_msg = CANMessage(
            timestamp=t,
            can_id=BMS_STATUS_ID,
            data=status_data
        )
        can_log.append(status_msg)
        
        # VCU processes message and decides state
        current_vcu_state = vcu_state_from_frame(status_data, cooling_cmd[k+1])
        vcu_states[k+1] = current_vcu_state.value
        
        # Send detail frame (less frequently - every 5 seconds)
        if int(t) % 5 == 0:
            soh = 1.0 - (cycle_count / 1000.0) * 0.2  # Simple SOH decay
            detail_data = encode_bms_detail_frame(
                v_cell_min=v_cell_min[k+1],
                v_cell_max=v_cell_max[k+1],
                soh=max(0.8, soh),
                cycles=cycle_count
            )
            
            detail_msg = CANMessage(
                timestamp=t,
                can_id=BMS_DETAIL_ID,
                data=detail_data
            )
            can_log.append(detail_msg)
    else:
        vcu_states[k+1] = vcu_states[k]

# Final step
i_cmd[-1] = current_profile(time[-1])
i_pack[-1] = 0.0 if shutdown_cmd[-1] else i_cmd[-1]

# ============================================================
# 10. Advanced visualization
# ============================================================

fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(7, 2, hspace=0.4, wspace=0.3)

# Current profile
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, i_cmd, 'b-', label='I_cmd', linewidth=1.5, alpha=0.7)
ax1.plot(time, i_pack, 'r--', label='I_pack (after shutdown)', linewidth=2)
ax1.axhline(I_OC, color='red', linestyle=':', alpha=0.7, label='Overcurrent limit')
ax1.axhline(-I_OC, color='red', linestyle=':', alpha=0.7)
ax1.axhline(I_WARN, color='orange', linestyle=':', alpha=0.5)
ax1.axhline(-I_WARN, color='orange', linestyle=':', alpha=0.5)
ax1.set_ylabel("Current [A]", fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("Pack Current Profile", fontweight='bold', fontsize=11)

# Pack voltage
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time, v_pack_arr, 'g-', linewidth=1.5)
v_ov = V_OV_CELL * N_SERIES
v_uv = V_UV_CELL * N_SERIES
ax2.axhline(v_ov, color='red', linestyle=':', label='OV limit')
ax2.axhline(v_uv, color='red', linestyle=':', label='UV limit')
ax2.set_ylabel("Pack Voltage [V]", fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("Pack Terminal Voltage", fontweight='bold', fontsize=11)

# SOC comparison
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(time, soc_coulomb * 100, 'b-', linewidth=1, alpha=0.6, label='Coulomb counting')
ax3.plot(time, soc_voltage * 100, 'r-', linewidth=1, alpha=0.6, label='Voltage-based')
ax3.plot(time, soc_blended * 100, 'g-', linewidth=2, label='Blended (80/20)')
ax3.set_ylabel("SOC [%]", fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title("State of Charge Estimation", fontweight='bold', fontsize=11)

# Temperature
ax4 = fig.add_subplot(gs[3, :])
ax4.plot(time, t_pack, 'm-', linewidth=1.5)
ax4.axhline(T_OV, color='red', linestyle=':', label='Overtemp fault')
ax4.axhline(T_WARN, color='orange', linestyle=':', label='Temp warning')
ax4.axhline(T_COOL_ON, color='blue', linestyle=':', label='Cooling on')
ax4.axhline(T_COOL_OFF, color='cyan', linestyle=':', alpha=0.7, label='Cooling off')
ax4.fill_between(time, 0, t_pack, where=cooling_cmd, alpha=0.2, color='blue', label='Cooling active')
ax4.set_ylabel("Temperature [°C]", fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_title("Pack Temperature & Cooling", fontweight='bold', fontsize=11)

# Cell imbalance
ax5 = fig.add_subplot(gs[4, :])
ax5.plot(time, v_cell_min, 'b-', linewidth=1, alpha=0.7, label='Min cell voltage')
ax5.plot(time, v_cell_max, 'r-', linewidth=1, alpha=0.7, label='Max cell voltage')
imbalance_mv = (v_cell_max - v_cell_min) * 1000
ax5_twin = ax5.twinx()
ax5_twin.plot(time, imbalance_mv, 'g--', linewidth=1, alpha=0.5, label='Imbalance')
ax5.set_ylabel("Cell Voltage [V]", fontweight='bold')
ax5_twin.set_ylabel("Imbalance [mV]", fontweight='bold', color='g')
ax5.legend(loc='upper left', fontsize=9)
ax5_twin.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_title("Cell Voltage Balance", fontweight='bold', fontsize=11)

# Fault flags
ax6 = fig.add_subplot(gs[5, :])
fault_any = fault_overV | fault_underV | fault_overT | fault_underT | fault_overI
ax6.fill_between(time, 0, fault_any.astype(int), alpha=0.3, color='red', label='Fault active')
ax6.plot(time, shutdown_cmd.astype(int), 'r-', linewidth=2, label='Shutdown')
ax6.plot(time, cooling_cmd.astype(int), 'b-', linewidth=2, label='Cooling')
ax6.set_ylabel("Control Flags", fontweight='bold')
ax6.set_ylim([-0.1, 1.3])
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_title("Fault & Control Signals", fontweight='bold', fontsize=11)

# VCU state
ax7 = fig.add_subplot(gs[6, :])
ax7.plot(time, vcu_states, 'k-', linewidth=2, drawstyle='steps-post')
ax7.set_ylabel("VCU State", fontweight='bold')
ax7.set_xlabel("Time [s]", fontweight='bold')
ax7.set_yticks([0, 1, 2, 3, 4])
ax7.set_yticklabels(['NORMAL', 'WARNING', 'COOLING', 'SHUTDOWN', 'FAULT_LATCHED'], fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_title("VCU State Machine", fontweight='bold', fontsize=11)

plt.savefig('bms_can_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 11. CAN log analysis & export
# ============================================================

print("\n" + "="*80)
print(" BMS CAN SIMULATION SUMMARY")
print("="*80)
print(f"Simulation Duration:      {T_END:.1f} seconds")
print(f"CAN Message Period:       {CAN_PERIOD:.1f} seconds")
print(f"Total CAN Messages:       {len(can_log)}")
print(f"Pack Configuration:       {N_SERIES}s{N_PARALLEL}p ({C_PACK_AH:.1f} Ah)")

print(f"\n" + "-"*80)
print(" FINAL STATE")
print("-"*80)
print(f"SOC (blended):            {soc_blended[-1]*100:.2f}%")
print(f"Pack Voltage:             {v_pack_arr[-1]:.3f} V")
print(f"Temperature:              {t_pack[-1]:.2f} °C")
print(f"VCU State:                {VCUState(vcu_states[-1]).name}")
print(f"Shutdown Active:          {'YES' if shutdown_cmd[-1] else 'NO'}")
print(f"Cooling Active:           {'YES' if cooling_cmd[-1] else 'NO'}")

print(f"\n" + "-"*80)
print(" OPERATIONAL STATISTICS")
print("-"*80)
print(f"Max Temperature:          {np.max(t_pack):.2f} °C")
print(f"Min Pack Voltage:         {np.min(v_pack_arr):.3f} V")
print(f"Max Pack Voltage:         {np.max(v_pack_arr):.3f} V")
print(f"Max Current (abs):        {np.max(np.abs(i_pack)):.2f} A")
print(f"Max Cell Imbalance:       {np.max((v_cell_max - v_cell_min) * 1000):.1f} mV")

print(f"\n" + "-"*80)
print(" CAN MESSAGE BREAKDOWN")
print("-"*80)
status_msgs = [m for m in can_log if m.can_id == BMS_STATUS_ID]
detail_msgs = [m for m in can_log if m.can_id == BMS_DETAIL_ID]
print(f"Status Messages (0x100):  {len(status_msgs)}")
print(f"Detail Messages (0x101):  {len(detail_msgs)}")

print(f"\n" + "-"*80)
print(" SAMPLE CAN MESSAGES")
print("-"*80)

for i, msg in enumerate(can_log[:5]):
    print(f"\nMessage #{i+1} @ t={msg.timestamp:.3f}s")
    print(f"  CAN ID:    0x{msg.can_id:03X}")
    print(f"  DLC:       {msg.dlc}")
    print(f"  Data:      {msg.to_hex_string()}")
    
    if msg.can_id == BMS_STATUS_ID:
        decoded = decode_bms_status_frame(msg.data)
        print(f"  Decoded:")
        print(f"    Voltage:     {decoded['voltage']:.2f} V")
        print(f"    Current:     {decoded['current']:.2f} A")
        print(f"    SOC:         {decoded['soc']}%")
        print(f"    Temperature: {decoded['temperature']:.1f} °C")
        print(f"    Faults:      0x{decoded['faults']:02X}")
        print(f"    Checksum:    {'VALID' if decoded['checksum_valid'] else 'INVALID'}")

print(f"\n" + "-"*80)
print(" FAULT OCCURRENCE")
print("-"*80)
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

# Export CAN log to JSON
can_log_file = 'bms_can_messages.json'
with open(can_log_file, 'w') as f:
    json.dump([msg.to_dict() for msg in can_log], f, indent=2)
print(f"✓ CAN message log exported to: {can_log_file}")

# Export decoded CAN data for analysis
decoded_log = []
for msg in can_log:
    if msg.can_id == BMS_STATUS_ID:
        decoded = decode_bms_status_frame(msg.data)
        decoded_log.append({
            'timestamp': msg.timestamp,
            'voltage': decoded['voltage'],
            'current': decoded['current'],
            'soc': decoded['soc'],
            'temperature': decoded['temperature'],
            'faults': decoded['faults'],
            'checksum_valid': decoded['checksum_valid']
        })

decoded_file = 'bms_decoded_data.json'
with open(decoded_file, 'w') as f:
    json.dump(decoded_log, f, indent=2)
print(f"✓ Decoded CAN data exported to: {decoded_file}")

# Export simulation summary
summary_file = 'bms_can_summary.txt'
with open(summary_file, 'w') as f:
    f.write("BMS CAN SIMULATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Configuration: {N_SERIES}s{N_PARALLEL}p Li-ion pack ({C_PACK_AH:.1f} Ah)\n")
    f.write(f"Simulation time: {T_END:.1f}s\n")
    f.write(f"CAN period: {CAN_PERIOD:.1f}s\n\n")
    f.write(f"Final SOC: {soc_blended[-1]*100:.2f}%\n")
    f.write(f"Max temperature: {np.max(t_pack):.2f}°C\n")
    f.write(f"Total CAN messages: {len(can_log)}\n")
    f.write(f"Total faults: {sum(fault_counts.values())}\n\n")
    f.write("Fault breakdown:\n")
    for fname, fcount in fault_counts.items():
        if fcount > 0:
            f.write(f"  {fname}: {fcount} occurrences\n")
print(f"✓ Summary exported to: {summary_file}")

# Create CAN trace file (Vector CANalyzer format)
trace_file = 'bms_can_trace.asc'
with open(trace_file, 'w') as f:
    f.write("date Mon Jan 01 00:00:00 2024\n")
    f.write("base hex  timestamps absolute\n")
    f.write("internal events logged\n")
    f.write("Begin Triggerblock Mon Jan 01 00:00:00 2024\n")
    f.write("   0.000000 Start of measurement\n")
    
    for msg in can_log:
        # Format: timestamp CAN channel ID Rx/Tx dlc data0 data1 ... data7
        f.write(f"   {msg.timestamp:>10.6f} 1  {msg.can_id:03X}             Rx   d {msg.dlc} ")
        f.write(" ".join([f"{b:02X}" for b in msg.data]))
        f.write("\n")
    
    f.write("End Triggerblock\n")
print(f"✓ CAN trace exported to: {trace_file} (Vector format)")

print("\n" + "="*80)
print(" EXPORT COMPLETE")
print("="*80)
print("\nGenerated files:")
print(f"  1. bms_can_simulation.png     - Comprehensive plots")
print(f"  2. {can_log_file}              - Raw CAN messages (JSON)")
print(f"  3. {decoded_file}              - Decoded telemetry (JSON)")
print(f"  4. {summary_file}              - Text summary")
print(f"  5. {trace_file}                - CAN trace (Vector format)")
print("\n" + "="*80 + "\n")