import asyncio
import json
import os
import subprocess
import time
import urllib.request
import base64
import websockets

cdp_port = 9225
url = "http://localhost:8501"
artifact_path = r"C:\Users\Tesse\.gemini\antigravity\brain\15be300f-9ff1-4935-9c1f-44889c5e0d75\reproduce_browser.png"

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

async def run_reproduction():
    print("Connecting to Edge CDP on port 9225...")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://localhost:{cdp_port}/json") as resp:
                targets = json.loads(resp.read().decode())
                page_target = next((t for t in targets if t.get("type") == "page"), None)
                if page_target and "webSocketDebuggerUrl" in page_target:
                    ws_url = page_target["webSocketDebuggerUrl"]
                    print(f"Connecting to ws: {ws_url}")
                    async with websockets.connect(ws_url) as ws:
                        # Wait for initial load
                        for poll in range(25):
                            eval_cmd = {
                                "id": 10 + poll,
                                "method": "Runtime.evaluate",
                                "params": {"expression": "document.querySelector('textarea') !== null"}
                            }
                            await ws.send(json.dumps(eval_cmd))
                            eval_res = json.loads(await ws.recv())
                            if eval_res.get("result", {}).get("result", {}).get("value", False):
                                print(f"Page loaded on poll {poll}!")
                                break
                            await asyncio.sleep(1)

                        await asyncio.sleep(2)

                        # Inspect initial Recent Questions
                        eval_initial = {
                            "id": 100,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": """
                                (() => {
                                    const sidebar = document.querySelector('section[data-testid="stSidebar"]');
                                    return sidebar ? sidebar.innerText : "no sidebar";
                                })()
                                """
                            }
                        }
                        await ws.send(json.dumps(eval_initial))
                        res_initial = json.loads(await ws.recv())
                        print("Initial sidebar text:\n", res_initial.get("result", {}).get("result", {}).get("value"))

                        # Expand Recent Questions in sidebar
                        eval_expand = {
                            "id": 101,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": """
                                (() => {
                                    const exp = document.querySelector('section[data-testid="stSidebar"] [data-testid="stExpander"] details');
                                    if (exp) { exp.open = true; return "opened expander"; }
                                    return "expander not found";
                                })()
                                """
                            }
                        }
                        await ws.send(json.dumps(eval_expand))
                        res_exp = json.loads(await ws.recv())
                        print("Expander status:", res_exp.get("result", {}).get("result", {}).get("value"))
                        await asyncio.sleep(1)

                        # Check text inside expander
                        eval_exp_text = {
                            "id": 102,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": """
                                (() => {
                                    const exp = document.querySelector('section[data-testid="stSidebar"] [data-testid="stExpander"]');
                                    return exp ? exp.innerText : "no expander";
                                })()
                                """
                            }
                        }
                        await ws.send(json.dumps(eval_exp_text))
                        res_exp_text = json.loads(await ws.recv())
                        print("Expander content:\n", res_exp_text.get("result", {}).get("result", {}).get("value"))

                        # Take screenshot
                        cmd_ss = {
                            "id": 999,
                            "method": "Page.captureScreenshot",
                            "params": {"format": "png"}
                        }
                        await ws.send(json.dumps(cmd_ss))
                        resp_str = await ws.recv()
                        resp_data = json.loads(resp_str)
                        png_bytes = base64.b64decode(resp_data["result"]["data"])
                        with open(artifact_path, "wb") as f:
                            f.write(png_bytes)
                        print(f"Saved initial reproduction screenshot to {artifact_path}")
                        return
        except Exception as e:
            await asyncio.sleep(1)

try:
    asyncio.run(run_reproduction())
finally:
    edge_proc.terminate()
    try:
        edge_proc.wait(timeout=3)
    except Exception:
        edge_proc.kill()
    print("Cleaned up.")

