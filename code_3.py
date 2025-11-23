import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Simulation configuration
# ============================================================

DT = 0.1            # time step [s]
T_END = 1000.0      # total simulation time [s]
N_STEPS = int(T_END / DT) + 1

time = np.linspace(0.0, T_END, N_STEPS)

# CAN send period (e.g. 1 message per second)
CAN_PERIOD = 1.0

# CAN message ID for BMS status
BMS_STATUS_ID = 0x100

# ============================================================
# 2. Cell & pack parameters (2s2p Li-ion pack)
# ============================================================

# Single cell assumptions (simplified Li-ion)
V_MIN_CELL = 3.0        # V at 0% SOC
V_MAX_CELL = 4.2        # V at 100% SOC
C_CELL_AH  = 2.5        # cell capacity [Ah]
R_CELL     = 0.05       # internal resistance per cell [ohm]

# Pack configuration: 2 in series, 2 in parallel
N_SERIES   = 2
N_PARALLEL = 2

C_PACK_AH  = C_CELL_AH * N_PARALLEL         # Ah
R_STRING   = R_CELL * N_SERIES              # two in series
R_PACK     = R_STRING / N_PARALLEL          # two strings in parallel

# Capacity in Coulombs (for SOC update)
C_PACK_C   = C_PACK_AH * 3600.0             # [Coulomb]

# ============================================================
# 3. Thermal model parameters
# ============================================================

T_AMB       = 25.0     # ambient temperature [°C]
T_INIT      = 25.0     # initial pack temperature [°C]
K_HEAT      = 0.001    # heating gain (I^2 * R term)
K_COOL      = 0.0005   # cooling gain

# ============================================================
# 4. Protection thresholds
# ============================================================

V_OV_CELL = 4.25   # overvoltage [V]
V_UV_CELL = 2.90   # undervoltage [V]

T_OV      = 60.0   # overtemperature [°C]
T_UV      = 0.0    # undertemperature [°C]

I_OC      = 12.0   # overcurrent threshold [A]

# Cooling hysteresis
T_COOL_ON  = 50.0  # turn cooling on above this
T_COOL_OFF = 45.0  # turn cooling off below this

# ============================================================
# 5. Helper: current profile I_cmd(t)
# ============================================================

def current_profile(t: float) -> float:
    """
    Commanded pack current as a function of time.
    Positive -> discharge, Negative -> charge.
    """
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
# 6. CAN helper functions
# ============================================================

def encode_signed_16(value_decitenth: int) -> (int, int):
    """
    Encode a signed 16-bit integer (e.g. current*10) into low, high bytes
    using two's complement.
    """
    if value_decitenth < 0:
        value_decitenth = (1 << 16) + value_decitenth
    value_decitenth &= 0xFFFF
    lo = value_decitenth & 0xFF
    hi = (value_decitenth >> 8) & 0xFF
    return lo, hi

def decode_signed_16(lo: int, hi: int) -> int:
    """
    Decode a signed 16-bit integer from low, high bytes.
    """
    raw = lo | (hi << 8)
    if raw & 0x8000:  # negative
        raw -= 1 << 16
    return raw

def encode_bms_can_frame(v_pack, i_pack, soc_frac, faults, temp_c):
    """
    Build an 8-byte CAN data field with:
    Byte 0-1: pack voltage [0.1 V]
    Byte 2-3: pack current [0.1 A, signed]
    Byte 4  : SOC [%]
    Byte 5  : fault bitfield
    Byte 6  : temperature [°C] (clipped 0..255)
    Byte 7  : reserved
    """
    # Voltage in 0.1 V units
    v_decivolt = int(round(v_pack * 10.0))
    v_lo = v_decivolt & 0xFF
    v_hi = (v_decivolt >> 8) & 0xFF

    # Current in 0.1 A units, signed
    i_decitenth = int(round(i_pack * 10.0))
    i_lo, i_hi = encode_signed_16(i_decitenth)

    # SOC in %
    soc_pct = int(round(np.clip(soc_frac * 100.0, 0.0, 100.0)))

    # Temperature
    temp_enc = int(round(np.clip(temp_c, 0.0, 255.0)))

    data = [
        v_lo, v_hi,
        i_lo, i_hi,
        soc_pct & 0xFF,
        faults & 0xFF,
        temp_enc & 0xFF,
        0x00  # reserved
    ]
    return data

def decode_bms_can_frame(data):
    """
    Decode the BMS CAN data back to physical values.
    Returns: (v_pack, i_pack, soc_pct, fault_bits, temp_c)
    """
    v_raw = data[0] | (data[1] << 8)
    v_pack = v_raw / 10.0

    i_raw = decode_signed_16(data[2], data[3])
    i_pack = i_raw / 10.0

    soc_pct = data[4]
    fault_bits = data[5]
    temp_c = data[6]

    return v_pack, i_pack, soc_pct, fault_bits, temp_c

def vcu_state_from_frame(data):
    """
    Simple VCU logic:
      - If any fault bits set -> SHUTDOWN
      - Else if temperature > threshold -> COOLING
      - Else -> NORMAL
    """
    v_pack, i_pack, soc_pct, fault_bits, temp_c = decode_bms_can_frame(data)

    if fault_bits != 0:
        return "SHUTDOWN"
    elif temp_c > T_COOL_ON:
        return "COOLING"
    else:
        return "NORMAL"

# ============================================================
# 7. State arrays
# ============================================================

soc        = np.zeros(N_STEPS)    # SOC [0..1]
v_pack_arr = np.zeros(N_STEPS)    # pack voltage [V]
t_pack     = np.zeros(N_STEPS)    # pack temperature [°C]
i_cmd      = np.zeros(N_STEPS)    # commanded current [A]
i_pack     = np.zeros(N_STEPS)    # actual current [A] (after shutdown)

# Fault flags & commands
fault_overV   = np.zeros(N_STEPS, dtype=bool)
fault_underV  = np.zeros(N_STEPS, dtype=bool)
fault_overT   = np.zeros(N_STEPS, dtype=bool)
fault_underT  = np.zeros(N_STEPS, dtype=bool)
fault_overI   = np.zeros(N_STEPS, dtype=bool)

shutdown_cmd  = np.zeros(N_STEPS, dtype=bool)
cooling_cmd   = np.zeros(N_STEPS, dtype=bool)

# CAN log: list of dicts
can_log = []

# Initial conditions
soc[0]        = 1.0       # 100% SOC
t_pack[0]     = T_INIT
last_can_time = -CAN_PERIOD  # force immediate send at t=0

# ============================================================
# 8. Core simulation loop
# ============================================================

for k in range(N_STEPS - 1):
    t = time[k]

    # 1) Commanded current from profile
    I_ref = current_profile(t)
    i_cmd[k] = I_ref

    # 2) Apply shutdown logic: if shutdown already active, I=0
    if shutdown_cmd[k]:
        I = 0.0
    else:
        I = I_ref
    i_pack[k] = I

    # 3) SOC update using coulomb counting
    delta_Q = I * DT
    soc[k+1] = soc[k] - delta_Q / C_PACK_C
    soc[k+1] = max(0.0, min(1.0, soc[k+1]))

    # 4) Open-circuit voltage per cell (linear OCV-SOC)
    voc_cell = V_MIN_CELL + (V_MAX_CELL - V_MIN_CELL) * soc[k+1]
    voc_pack = N_SERIES * voc_cell

    # 5) Terminal voltage with ohmic drop
    v_pack = voc_pack - I * R_PACK
    v_pack_arr[k+1] = v_pack

    # Estimate per-cell voltage
    v_cell_est = v_pack / N_SERIES

    # 6) Temperature update (simple lumped model)
    q_gen   = K_HEAT * (I ** 2) * R_PACK
    q_cool  = K_COOL * (t_pack[k] - T_AMB)
    t_next  = t_pack[k] + (q_gen - q_cool) * DT
    t_pack[k+1] = t_next

    # 7) Fault detection
    overV  = v_cell_est > V_OV_CELL
    underV = v_cell_est < V_UV_CELL
    overT  = t_next > T_OV
    underT = t_next < T_UV
    overI  = abs(I) > I_OC

    fault_overV[k+1]  = overV
    fault_underV[k+1] = underV
    fault_overT[k+1]  = overT
    fault_underT[k+1] = underT
    fault_overI[k+1]  = overI

    # 8) Shutdown command (severe faults)
    if overV or overT or overI:
        shutdown_cmd[k+1] = True
    else:
        shutdown_cmd[k+1] = False  # or latch: = shutdown_cmd[k]

    # 9) Cooling command with hysteresis
    if t_next > T_COOL_ON:
        cooling_cmd[k+1] = True
    elif t_next < T_COOL_OFF:
        cooling_cmd[k+1] = False
    else:
        cooling_cmd[k+1] = cooling_cmd[k]

    # 10) Build fault bitfield for CAN
    fault_bits = (
        (1 if overV  else 0) << 0 |
        (1 if underV else 0) << 1 |
        (1 if overT  else 0) << 2 |
        (1 if underT else 0) << 3 |
        (1 if overI  else 0) << 4
    )

    # 11) CAN transmission at fixed period
    if t - last_can_time >= CAN_PERIOD:
        last_can_time = t
        data = encode_bms_can_frame(
            v_pack=v_pack,
            i_pack=I,
            soc_frac=soc[k+1],
            faults=fault_bits,
            temp_c=t_next
        )
        # VCU receives and decides state
        vcu_state = vcu_state_from_frame(data)

        can_log.append({
            "time": t,
            "id": BMS_STATUS_ID,
            "data": data,
            "vcu_state": vcu_state
        })

# Final step currents
i_cmd[-1]  = current_profile(time[-1])
i_pack[-1] = 0.0 if shutdown_cmd[-1] else i_cmd[-1]

# ============================================================
# 9. Plots
# ============================================================

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

axs[0].plot(time, i_cmd, label="I_cmd")
axs[0].plot(time, i_pack, linestyle="--", label="I_pack")
axs[0].set_ylabel("Current [A]")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time, v_pack_arr)
axs[1].set_ylabel("Pack Voltage [V]")
axs[1].grid(True)

axs[2].plot(time, soc * 100.0)
axs[2].set_ylabel("SOC [%]")
axs[2].grid(True)

axs[3].plot(time, t_pack)
axs[3].set_ylabel("Temperature [°C]")
axs[3].grid(True)

fault_any = fault_overV | fault_underV | fault_overT | fault_underT | fault_overI
axs[4].plot(time, fault_any.astype(int), label="fault_any")
axs[4].plot(time, shutdown_cmd.astype(int), label="shutdown_cmd")
axs[4].plot(time, cooling_cmd.astype(int), label="cooling_cmd")
axs[4].set_ylabel("Flags")
axs[4].set_xlabel("Time [s]")
axs[4].legend()
axs[4].grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 10. Print a small CAN log sample
# ============================================================

print("Sample CAN messages:")
for entry in can_log[:10]:
    print(
        f"t={entry['time']:.1f}s  ID=0x{entry['id']:03X}  "
        f"data={entry['data']}  VCU_state={entry['vcu_state']}"
    )
