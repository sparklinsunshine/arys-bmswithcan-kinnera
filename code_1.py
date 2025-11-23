import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

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

# ===============================
# 4. Safety thresholds
# ===============================

V_OVER_THRESHOLD = 8.6      # overvoltage per pack
V_UNDER_THRESHOLD = 5.8     # undervoltage per pack
T_OVER_THRESHOLD = 45.0     # overtemperature
T_UNDER_THRESHOLD = 0.0     # undertemperature
I_OVER_THRESHOLD = 12.0     # overcurrent (absolute)
CELL_IMBALANCE_THRESHOLD = 0.15  # voltage difference threshold

# ===============================
# 5. VCU States
# ===============================

class VCUState(Enum):
    NORMAL = 0
    COOLING = 1
    SHUTDOWN = 2

@dataclass
class FaultFlags:
    overvoltage: bool = False
    undervoltage: bool = False
    overtemperature: bool = False
    undertemperature: bool = False
    overcurrent: bool = False
    cell_imbalance: bool = False
    
    def any_fault(self) -> bool:
        return any([self.overvoltage, self.undervoltage, 
                   self.overtemperature, self.undertemperature,
                   self.overcurrent, self.cell_imbalance])
    
    def to_byte(self) -> int:
        """Pack fault flags into a single byte"""
        return (int(self.overvoltage) << 0 |
                int(self.undervoltage) << 1 |
                int(self.overtemperature) << 2 |
                int(self.undertemperature) << 3 |
                int(self.overcurrent) << 4 |
                int(self.cell_imbalance) << 5)

@dataclass
class CANFrame:
    """Simplified CAN frame for BMS data"""
    timestamp: float
    pack_voltage: float  # V (scaled to uint16, 0.01V resolution)
    pack_current: float  # A (scaled to int16, 0.1A resolution)
    soc: float          # % (scaled to uint8, 0-100)
    temperature: float  # °C (scaled to int8, 1°C resolution)
    fault_flags: int    # byte with fault bits
    vcu_state: int      # VCU state enum value
    
    def encode(self) -> bytes:
        """Encode to CAN data bytes (simplified 8-byte payload)"""
        v_scaled = int(self.pack_voltage * 100) & 0xFFFF
        i_scaled = int(self.pack_current * 10) & 0xFFFF
        soc_scaled = int(self.soc) & 0xFF
        t_scaled = int(self.temperature) & 0xFF
        
        return bytes([
            (v_scaled >> 8) & 0xFF,
            v_scaled & 0xFF,
            (i_scaled >> 8) & 0xFF,
            i_scaled & 0xFF,
            soc_scaled,
            t_scaled,
            self.fault_flags,
            self.vcu_state
        ])

# ===============================
# 6. Helper functions
# ===============================

def current_profile(t: float) -> float:
    """Define pack current as a function of time"""
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

def voltage_to_soc(voltage: float, n_series: int = N_SERIES) -> float:
    """Estimate SOC from voltage (inverse of OCV curve)"""
    v_cell = voltage / n_series
    soc_est = (v_cell - V_MIN_CELL) / (V_MAX_CELL - V_MIN_CELL)
    return max(0.0, min(1.0, soc_est))

def detect_faults(v_pack: float, i_pack: float, t_pack: float, 
                 cell_voltages: List[float]) -> FaultFlags:
    """Detect various fault conditions"""
    faults = FaultFlags()
    
    faults.overvoltage = v_pack > V_OVER_THRESHOLD
    faults.undervoltage = v_pack < V_UNDER_THRESHOLD
    faults.overtemperature = t_pack > T_OVER_THRESHOLD
    faults.undertemperature = t_pack < T_UNDER_THRESHOLD
    faults.overcurrent = abs(i_pack) > I_OVER_THRESHOLD
    
    # Check cell imbalance
    if len(cell_voltages) > 1:
        v_max = max(cell_voltages)
        v_min = min(cell_voltages)
        faults.cell_imbalance = (v_max - v_min) > CELL_IMBALANCE_THRESHOLD
    
    return faults

def vcu_decide_state(faults: FaultFlags, temp: float) -> VCUState:
    """VCU decision logic based on faults and temperature"""
    # Critical faults trigger shutdown
    if (faults.overvoltage or faults.undervoltage or 
        faults.overcurrent or faults.cell_imbalance):
        return VCUState.SHUTDOWN
    
    # Temperature faults trigger appropriate response
    if faults.overtemperature:
        return VCUState.COOLING
    
    if faults.undertemperature:
        return VCUState.SHUTDOWN
    
    # High temperature (but not fault) triggers cooling
    if temp > 35.0:
        return VCUState.COOLING
    
    return VCUState.NORMAL

# ===============================
# 7. State arrays
# ===============================

soc_coulomb = np.zeros(N_STEPS)  # Coulomb counting SOC
soc_voltage = np.zeros(N_STEPS)  # Voltage-based SOC
soc_blended = np.zeros(N_STEPS)  # Blended SOC estimate
v_pack = np.zeros(N_STEPS)
t_pack = np.zeros(N_STEPS)
i_pack = np.zeros(N_STEPS)

# Cell-level voltages (for imbalance detection)
v_cell_1 = np.zeros(N_STEPS)
v_cell_2 = np.zeros(N_STEPS)

# Fault and state tracking
fault_log = []
vcu_states = np.zeros(N_STEPS, dtype=int)
can_frames = []

# Initial conditions
soc_coulomb[0] = 1.0
soc_voltage[0] = 1.0
soc_blended[0] = 1.0
t_pack[0] = T_INIT

# ===============================
# 8. Core simulation loop
# ===============================

ALPHA_BLEND = 0.7  # Weighting for coulomb counting (0.7) vs voltage (0.3)

for k in range(N_STEPS - 1):
    t = time[k]
    I = current_profile(t)
    i_pack[k] = I
    
    # SOC update - Coulomb counting
    delta_Q = I * DT
    soc_coulomb[k+1] = max(0.0, min(1.0, soc_coulomb[k] - delta_Q / C_PACK_C))
    
    # Open-circuit voltage and terminal voltage
    voc_cell = V_MIN_CELL + (V_MAX_CELL - V_MIN_CELL) * soc_coulomb[k+1]
    voc_pack = N_SERIES * voc_cell
    v_pack[k+1] = voc_pack - I * R_PACK
    
    # SOC estimate from voltage
    soc_voltage[k+1] = voltage_to_soc(v_pack[k+1])
    
    # Blended SOC (weighted average)
    soc_blended[k+1] = (ALPHA_BLEND * soc_coulomb[k+1] + 
                        (1 - ALPHA_BLEND) * soc_voltage[k+1])
    
    # Temperature update
    q_gen = K_HEAT * (I ** 2) * R_PACK
    q_cool = K_COOL * (t_pack[k] - T_AMB)
    t_pack[k+1] = t_pack[k] + (q_gen - q_cool) * DT
    
    # Model individual cell voltages (with slight imbalance)
    # Add small drift to simulate imbalance
    imbalance_factor = 0.02 * np.sin(t / 100.0)
    v_cell_1[k+1] = voc_cell * (1 + imbalance_factor) - (I / N_PARALLEL) * R_CELL
    v_cell_2[k+1] = voc_cell * (1 - imbalance_factor) - (I / N_PARALLEL) * R_CELL
    
    # Fault detection
    cell_voltages = [v_cell_1[k+1], v_cell_2[k+1]]
    faults = detect_faults(v_pack[k+1], I, t_pack[k+1], cell_voltages)
    fault_log.append((t, faults))
    
    # VCU state decision
    vcu_state = vcu_decide_state(faults, t_pack[k+1])
    vcu_states[k+1] = vcu_state.value
    
    # Create CAN frame every 1 second
    if k % int(1.0 / DT) == 0:
        can_frame = CANFrame(
            timestamp=t,
            pack_voltage=v_pack[k+1],
            pack_current=I,
            soc=soc_blended[k+1] * 100.0,
            temperature=t_pack[k+1],
            fault_flags=faults.to_byte(),
            vcu_state=vcu_state.value
        )
        can_frames.append(can_frame)

# Final values
i_pack[-1] = current_profile(time[-1])

# ===============================
# 9. Plot results
# ===============================

fig, axs = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

# Current
axs[0].plot(time, i_pack, 'b-', linewidth=1.5)
axs[0].axhline(I_OVER_THRESHOLD, color='r', linestyle='--', label='Overcurrent threshold')
axs[0].axhline(-I_OVER_THRESHOLD, color='r', linestyle='--')
axs[0].set_ylabel("Current [A]")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Voltage
axs[1].plot(time, v_pack, 'g-', linewidth=1.5, label='Pack voltage')
axs[1].axhline(V_OVER_THRESHOLD, color='r', linestyle='--', label='Overvoltage')
axs[1].axhline(V_UNDER_THRESHOLD, color='orange', linestyle='--', label='Undervoltage')
axs[1].set_ylabel("Pack Voltage [V]")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# SOC comparison
axs[2].plot(time, soc_coulomb * 100.0, 'b-', linewidth=1, label='Coulomb counting', alpha=0.7)
axs[2].plot(time, soc_voltage * 100.0, 'r-', linewidth=1, label='Voltage-based', alpha=0.7)
axs[2].plot(time, soc_blended * 100.0, 'g-', linewidth=2, label='Blended estimate')
axs[2].set_ylabel("SOC [%]")
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Temperature
axs[3].plot(time, t_pack, 'm-', linewidth=1.5)
axs[3].axhline(T_OVER_THRESHOLD, color='r', linestyle='--', label='Overtemp threshold')
axs[3].axhline(35.0, color='orange', linestyle='--', label='Cooling trigger')
axs[3].set_ylabel("Temperature [°C]")
axs[3].legend()
axs[3].grid(True, alpha=0.3)

# Cell imbalance
axs[4].plot(time, v_cell_1, 'b-', linewidth=1, label='Cell 1', alpha=0.7)
axs[4].plot(time, v_cell_2, 'r-', linewidth=1, label='Cell 2', alpha=0.7)
axs[4].set_ylabel("Cell Voltage [V]")
axs[4].legend()
axs[4].grid(True, alpha=0.3)

# VCU state
axs[5].plot(time, vcu_states, 'k-', linewidth=2, drawstyle='steps-post')
axs[5].set_ylabel("VCU State")
axs[5].set_yticks([0, 1, 2])
axs[5].set_yticklabels(['NORMAL', 'COOLING', 'SHUTDOWN'])
axs[5].set_xlabel("Time [s]")
axs[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bms_simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# 10. Summary report
# ===============================

print("\n" + "="*60)
print("BMS SIMULATION SUMMARY")
print("="*60)
print(f"Simulation duration: {T_END} seconds")
print(f"Final SOC: {soc_blended[-1]*100:.1f}%")
print(f"Final pack voltage: {v_pack[-1]:.2f} V")
print(f"Final temperature: {t_pack[-1]:.1f} °C")
print(f"Max temperature reached: {np.max(t_pack):.1f} °C")
print(f"Min voltage reached: {np.min(v_pack):.2f} V")
print(f"Max current: {np.max(np.abs(i_pack)):.1f} A")

print("\n" + "-"*60)
print("FAULT SUMMARY")
print("-"*60)

fault_counts = {
    'overvoltage': 0,
    'undervoltage': 0,
    'overtemperature': 0,
    'undertemperature': 0,
    'overcurrent': 0,
    'cell_imbalance': 0
}

for t, faults in fault_log:
    if faults.overvoltage: fault_counts['overvoltage'] += 1
    if faults.undervoltage: fault_counts['undervoltage'] += 1
    if faults.overtemperature: fault_counts['overtemperature'] += 1
    if faults.undertemperature: fault_counts['undertemperature'] += 1
    if faults.overcurrent: fault_counts['overcurrent'] += 1
    if faults.cell_imbalance: fault_counts['cell_imbalance'] += 1

for fault_type, count in fault_counts.items():
    if count > 0:
        print(f"{fault_type.capitalize()}: {count} occurrences")

print("\n" + "-"*60)
print("VCU STATE SUMMARY")
print("-"*60)

state_counts = {
    VCUState.NORMAL.value: np.sum(vcu_states == VCUState.NORMAL.value),
    VCUState.COOLING.value: np.sum(vcu_states == VCUState.COOLING.value),
    VCUState.SHUTDOWN.value: np.sum(vcu_states == VCUState.SHUTDOWN.value)
}

for state_val, count in state_counts.items():
    state_name = VCUState(state_val).name
    percentage = (count / N_STEPS) * 100
    print(f"{state_name}: {count} steps ({percentage:.1f}%)")

print("\n" + "-"*60)
print(f"CAN FRAMES GENERATED: {len(can_frames)}")
print("-"*60)
print("Sample CAN frame (last):")
if can_frames:
    last_frame = can_frames[-1]
    print(f"  Timestamp: {last_frame.timestamp:.1f} s")
    print(f"  Voltage: {last_frame.pack_voltage:.2f} V")
    print(f"  Current: {last_frame.pack_current:.2f} A")
    print(f"  SOC: {last_frame.soc:.1f}%")
    print(f"  Temperature: {last_frame.temperature:.1f} °C")
    print(f"  Fault flags: 0x{last_frame.fault_flags:02X}")
    print(f"  VCU state: {VCUState(last_frame.vcu_state).name}")
    print(f"  Encoded bytes: {last_frame.encode().hex()}")

print("\n" + "="*60)