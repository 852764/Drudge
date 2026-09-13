"""Standalone Windows command owner, launched by absolute path with Python -I.

Only stdlib imports belong here. The owner joins a kill-on-close Job Object
BEFORE spawning the shell. Its non-inheritable job handle is closed by the OS
on exit (including TerminateProcess), killing all remaining descendants.
This avoids the shell-exited and child-spawn races of PID-tree enumeration.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess
import sys


def _own_job() -> int:
    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)  # NULL security attributes: not inheritable.
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel.CloseHandle(job)
        raise
    # Do not close while this process is alive: that would kill the owner too.
    return job


def main() -> int:
    if os.name != "nt" or len(sys.argv) != 2:
        print("Windows command owner requires exactly one command", file=sys.stderr, flush=True)
        return 125
    try:
        _own_job()
    except OSError as exc:
        print(f"Command not started: Windows Job Object setup failed: {exc}", file=sys.stderr, flush=True)
        return 125
    return subprocess.call(sys.argv[1], shell=True, executable=os.environ.get("COMSPEC", "cmd.exe"))


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"Command owner failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        code = 125
    # ExitProcess closes the job even if the shell left children holding pipes.
    # Preserve Windows unsigned exception exit codes through Python's signed int.
    os._exit(code if code < 2**31 else code - 2**32)
