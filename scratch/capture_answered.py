import os
import subprocess
import time
import urllib.request

port = 8519
url = f"http://localhost:{port}"
artifact_path = r"C:\Users\Tesse\.gemini\antigravity\brain\15be300f-9ff1-4935-9c1f-44889c5e0d75\phase3_answered_preview.png"

# Start streamlit
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.Popen(
    [
        r".venv\Scripts\python.exe",
        "-m",
        "streamlit",
        "run",
        r"scratch\mocked_app.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

try:
    # Wait for server ready
    ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        print("Server not ready in time.")
    else:
        print("Server ready, giving 3 seconds for initial render...")
        time.sleep(3)

        # Locate msedge
        edge_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        edge_exe = None
        for ep in edge_paths:
            if os.path.exists(ep):
                edge_exe = ep
                break

        if edge_exe:
            print("Capturing screenshot via Edge headless...")
            cmd = [
                edge_exe,
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--virtual-time-budget=8000",
                f"--screenshot={artifact_path}",
                "--window-size=1280,950",
                "--hide-scrollbars",
                url,
            ]
            subprocess.run(cmd, check=True)
            print(f"Screenshot successfully saved to {artifact_path}")
        else:
            print("Edge executable not found.")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("Streamlit process terminated.")
