import asyncio
import json
import os
import subprocess
import time
import urllib.request
import base64
import websockets

port = 8522
cdp_port = 9222
url = f"http://localhost:{port}"
artifact_path = r"C:\Users\Tesse\.gemini\antigravity\brain\15be300f-9ff1-4935-9c1f-44889c5e0d75\phase3_answered_preview.png"

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
st_proc = subprocess.Popen(
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

edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
edge_exe = next((ep for ep in edge_paths if os.path.exists(ep)), None)

edge_proc = subprocess.Popen(
    [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--remote-debugging-port={cdp_port}",
        "--window-size=1280,1050",
        url,
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

async def take_screenshot():
    print("Waiting for Streamlit and CDP...")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://localhost:{cdp_port}/json") as resp:
                targets = json.loads(resp.read().decode())
                page_target = next((t for t in targets if t.get("type") == "page" and ("8522" in t.get("url", "") or "devtools" in t.get("url", ""))), None)
                if not page_target:
                    page_target = next((t for t in targets if t.get("type") == "page"), None)
                if page_target and "webSocketDebuggerUrl" in page_target:
                    ws_url = page_target["webSocketDebuggerUrl"]
                    print(f"Connecting to CDP: {ws_url}")
                    async with websockets.connect(ws_url) as ws:
                        # Poll for DOM content
                        for poll in range(25):
                            eval_cmd = {
                                "id": 10 + poll,
                                "method": "Runtime.evaluate",
                                "params": {"expression": "document.querySelector('.saas-top-navbar') !== null && document.querySelector('.hero-main-title') !== null"}
                            }
                            await ws.send(json.dumps(eval_cmd))
                            eval_res = json.loads(await ws.recv())
                            is_ready = eval_res.get("result", {}).get("result", {}).get("value", False)
                            print(f"Poll {poll}: Streamlit UI ready = {is_ready}")
                            if is_ready:
                                break
                            await asyncio.sleep(1)

                        print("Waiting 2 extra seconds for animations/fonts to settle...")
                        await asyncio.sleep(2)

                        # Open Recent Questions expander in sidebar to visualize native buttons
                        open_sidebar_exp = {
                            "id": 998,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": """
                                const sidebarDetails = document.querySelector('section[data-testid="stSidebar"] [data-testid="stExpander"] details');
                                if (sidebarDetails) { sidebarDetails.open = true; }
                                """
                            }
                        }
                        await ws.send(json.dumps(open_sidebar_exp))
                        await ws.recv()
                        await asyncio.sleep(1)

                        # Capture screenshot
                        cmd = {
                            "id": 999,
                            "method": "Page.captureScreenshot",
                            "params": {"format": "png", "captureBeyondViewport": True}
                        }
                        await ws.send(json.dumps(cmd))
                        resp_str = await ws.recv()
                        resp_data = json.loads(resp_str)
                        png_bytes = base64.b64decode(resp_data["result"]["data"])
                        with open(artifact_path, "wb") as f:
                            f.write(png_bytes)
                        print(f"Saved screenshot to {artifact_path} ({len(png_bytes)} bytes)")
                        return
        except Exception as e:
            await asyncio.sleep(1)

try:
    asyncio.run(take_screenshot())
finally:
    edge_proc.terminate()
    st_proc.terminate()
    try:
        edge_proc.wait(timeout=3)
        st_proc.wait(timeout=3)
    except Exception:
        edge_proc.kill()
        st_proc.kill()
    print("Done.")
