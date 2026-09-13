from __future__ import annotations

import asyncio
import ctypes
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import _windows_job
from tools.context import ToolContext
from tools.output_capture import OutputCapture
from tools.terminal import terminal_handler, _terminate_process_tree


class ProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.context = ToolContext(workspace=self.root, enabled_toolsets=frozenset({"terminal"}), approval_mode="auto")

    def command(self, code):
        script = self.root / "command with spaces.py"
        script.write_text(code, encoding="utf-8")
        return f'"{sys.executable}" -u "{script}"'

    def tree_command(self, *, stay_alive):
        ready = self.root / "child-ready"
        child = f"import time; from pathlib import Path; Path({str(ready)!r}).touch(); print('TREE_READY', flush=True); time.sleep(5)"
        return self.command(
            "import subprocess, sys, time\nfrom pathlib import Path\n"
            f"child = subprocess.Popen([sys.executable, '-u', '-c', {child!r}])\n"
            "print('CHILD_PID=' + str(child.pid), flush=True)\n"
            "deadline = time.monotonic() + 3\n"
            f"while not Path({str(ready)!r}).exists() and time.monotonic() < deadline:\n    time.sleep(0.01)\n"
            + ("time.sleep(5)\n" if stay_alive else "print('SHELL_EXIT', flush=True)\n")
        )

    def assert_windows_child_exited(self, text):
        if os.name != "nt":
            return
        from ctypes import wintypes
        pid = int(text.split("CHILD_PID=", 1)[1].split()[0])
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE, no mutation.
        if not handle:
            self.assertEqual(ctypes.get_last_error(), 87)  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return
        try:
            self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0, "Owned descendant is still running")
        finally:
            kernel.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object lifecycle")
    def test_shell_exit_closes_job_and_descendants_holding_pipes(self):
        start = time.monotonic()
        result = json.loads(asyncio.run(terminal_handler(self.tree_command(stay_alive=False), timeout=3, context=self.context)))
        self.assertTrue(result["ok"], result)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn("SHELL_EXIT", result["content"])
        self.assert_windows_child_exited(result["content"])

    def test_timeout_stops_owned_descendants(self):
        result = json.loads(asyncio.run(terminal_handler(self.tree_command(stay_alive=True), timeout=1, context=self.context)))
        self.assertFalse(result["ok"])
        self.assertTrue(result["metadata"]["timed_out"])
        self.assertEqual(result["metadata"]["warnings"], ["Full stream output was not persisted; only a bounded preview is available."])
        self.assert_windows_child_exited(result["content"])

    def test_repeated_cancel_waits_for_cleanup_before_closing_captures(self):
        command = self.tree_command(stay_alive=True)
        captured = []
        write = OutputCapture.write
        close = OutputCapture.close

        async def exercise():
            ready, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
            cleanup_finished = False

            def observe(capture, text):
                write(capture, text)
                captured.append(text)
                if "TREE_READY" in "".join(captured):
                    ready.set()

            def close_after_cleanup(capture):
                self.assertTrue(cleanup_finished)
                close(capture)

            async def delayed_cleanup(process):
                nonlocal cleanup_finished
                entered.set()
                await release.wait()
                await _terminate_process_tree(process)
                cleanup_finished = True

            with patch.object(OutputCapture, "write", observe), patch.object(OutputCapture, "close", close_after_cleanup), patch("tools.terminal._terminate_process_tree", delayed_cleanup):
                task = asyncio.create_task(terminal_handler(command, context=self.context))
                try:
                    await asyncio.wait_for(ready.wait(), 3)
                    task.cancel()
                    await asyncio.wait_for(entered.wait(), 3)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
        asyncio.run(exercise())
        self.assert_windows_child_exited("".join(captured))

    def test_cleanup_errors_are_visible_in_result_metadata(self):
        async def cleanup_with_diagnostic(process):
            await _terminate_process_tree(process)
            raise OSError("cleanup diagnostic fixture")
        with patch("tools.terminal._terminate_process_tree", cleanup_with_diagnostic):
            result = json.loads(asyncio.run(terminal_handler(self.command("import time; time.sleep(5)"), timeout=1, context=self.context)))
        self.assertFalse(result["ok"])
        self.assertIn("cleanup diagnostic fixture", result["metadata"]["warnings"][0])

    def test_unavailable_spool_does_not_prevent_command_or_change_exit_status(self):
        from dataclasses import replace
        context = replace(self.context, save_tool_output=Mock())
        with patch("tools.output_capture.tempfile.TemporaryFile", side_effect=OSError("disk full")):
            result = json.loads(asyncio.run(terminal_handler(self.command("print('ran once')"), context=context)))
        self.assertTrue(result["ok"])
        self.assertIn("ran once", result["content"])
        self.assertIn("Output spool unavailable", result["metadata"]["warnings"][0])

    def test_job_setup_failure_does_not_run_command(self):
        with patch.object(_windows_job.os, "name", "nt"), patch.object(sys, "argv", ["runner", "echo test"]), patch.object(_windows_job, "_own_job", side_effect=OSError("fixture denied")), patch.object(_windows_job.subprocess, "call") as run, patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(_windows_job.main(), 125)
        run.assert_not_called()
        self.assertIn("Command not started", err.getvalue())


if __name__ == "__main__":
    unittest.main()
