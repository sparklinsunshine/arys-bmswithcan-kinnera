# BMS with CAN Simulation (2s2p Li-ion Pack)

This repository contains a Python-based simulation of a **Battery Management System (BMS)** for a simplified **2s2p Li-ion battery pack**, with:

- Pack electrical & thermal modeling  
- SOC estimation via coulomb counting (with voltage cross-check)  
- Fault detection (OV, UV, OT, UT, OC, optional imbalance)  
- Shutdown & cooling control logic  
- CAN message encoding/decoding  
- Simple VCU logic reacting to BMS CAN frames  

The project was developed as part of an EV-focused embedded systems assignment.

---

## 1. Features

### Battery Pack Model
- 2s2p pack (two cells in series, two such strings in parallel)
- Linear OCV–SOC model: 3.0 V → 4.2 V per cell
- Coulomb counting for SOC
- Lumped thermal model with I²R heating and ambient cooling

### BMS Logic
- Fault detection:
  - Overvoltage (OV)
  - Undervoltage (UV)
  - Overtemperature (OT)
  - Undertemperature (UT)
  - Overcurrent (OC)
  - (Optional) Imbalance flag
- Control actions:
  - Shutdown command (forces current to 0 A)
  - Cooling command with hysteresis
  - (Optional) Balancing command during charge + imbalance

### CAN & VCU
- Periodic BMS status frame at 1 Hz:
  - Pack voltage (0.1 V units)
  - Pack current (0.1 A units, signed)
  - SOC (%)
  - Fault bitfield
  - Temperature (°C)
- Simple VCU logic:
  - `NORMAL`
  - `COOLING`
  - `SHUTDOWN` (on critical faults)

---

## 2. Repository Structure

```text
.
├─ code_1.py          # Version 1: pack model only (V, I, SOC, T)
├─ code_2.py   # Version 2: adds fault detection & shutdown/cooling
├─ code_3.py         # Version 3: adds CAN encoding + VCU logic
├─ report/                # PDF report and figures
└─ README.md
