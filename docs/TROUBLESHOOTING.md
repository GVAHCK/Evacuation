# Platform Troubleshooting & Operational FAQ

## 1. Windows Smart App Control / WDAC DLL Restrictions
- **Symptom**: `ImportError: DLL load failed` or `Access Denied` when importing PyTorch or binary dependencies located inside OneDrive or Desktop folders.
- **Root Cause**: Windows 11 Smart App Control blocks unsigned `.pyd` and `.dll` files when loaded from user sync folders.
- **Resolution**: Use the provided launcher script `run_py.bat` or `run_py.ps1`. The launcher configures Python to execute from `AppData\Roaming\uv\python\` where all base binaries and signed site-packages run unrestricted.

---

## 2. PostgreSQL Connection Failure
- **Symptom**: `OperationalError: could not connect to server: Connection refused`.
- **Resolution**: The system automatically switches to the local SQLite engine (`evacuation.db`), logging a warning. If you wish to use PostgreSQL, start the container:
```bash
docker compose up -d postgres
```

---

## 3. Mobile Device Cannot Connect to Backend
- **Symptom**: `Network Error` or timeout when attempting to login on physical phone.
- **Resolution**:
  1. Ensure phone and host machine are on the same Wi-Fi subnet.
  2. Open Windows Firewall port 8000:
     ```powershell
     New-NetFirewallRule -DisplayName "FastAPI Evac" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
     ```
  3. In `mobile/src/screens/SettingsScreen.tsx`, set the backend URL to your PC's IP address (e.g. `http://192.168.1.100:8000`).
  4. In Android Studio Emulator, use `http://10.0.2.2:8000`.

---

## 4. SUMO Missing or TraCI Disconnect
- **Symptom**: `SUMO binary 'sumo' not found in PATH`.
- **Resolution**: Install SUMO via system package manager or set `SUMO_HOME` in `.env`. Note that `SUMOManager` operates in resilient mode: missing SUMO binaries will never halt the API or model inference.
