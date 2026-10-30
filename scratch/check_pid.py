import ctypes
from ctypes import wintypes
import os

# Query process working directory via Win32 NtQueryInformationProcess
# Or simply check handles / command line
print("Script running in:", os.path.abspath("."))
print("Checking PID 16364 and 14568")

# Let's inspect handles using PowerShell or handle.exe if available, or just inspect environment
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

kernel32 = ctypes.windll.kernel32
ntdll = ctypes.windll.ntdll

hProc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, 16364)
print("OpenProcess result:", bool(hProc))
if hProc:
    kernel32.CloseHandle(hProc)

