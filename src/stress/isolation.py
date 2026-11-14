"""Windows Job Object ownership and resource limits, using only the standard library."""
import ctypes
from ctypes import wintypes as W
import os
import subprocess


class BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', W.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                  'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimit), ('IoInfo', IOCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class ThreadEntry(ctypes.Structure):
    _fields_ = [('dwSize',W.DWORD),('cntUsage',W.DWORD),('th32ThreadID',W.DWORD),
                ('th32OwnerProcessID',W.DWORD),('tpBasePri',W.LONG),('tpDeltaPri',W.LONG),('dwFlags',W.DWORD)]


class Job:
    """Create suspended processes and assign them before resuming their first thread.

    This prevents venv launchers from spawning unowned grandchildren before assignment.
    Closing the handle kills every descendant, including hung HTTP calculations.
    """
    def __init__(self, memory_bytes):
        if os.name != 'nt':
            raise RuntimeError('The bounded campaign launcher currently requires Windows')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.restype = W.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
        self.kernel.CloseHandle.argtypes = [W.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.JobMemoryLimit = int(memory_bytes)
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self.children = []

    def start(self, python, module, args, *, cwd, env, stdout):
        gate = ('import sys,runpy; sys.stdin.buffer.read(1); '
                'sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0],run_name="__main__")')
        child = subprocess.Popen([str(python), '-u', '-c', gate, module, *map(str, args)],
                                 cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=stdout,
                                 stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW | 0x4)  # CREATE_SUSPENDED
        if not self.kernel.AssignProcessToJobObject(self.handle, W.HANDLE(child._handle)):
            child.kill()
            child.wait()
            raise ctypes.WinError(ctypes.get_last_error())
        self.children.append(child)
        try:
            self.resume(child.pid)
        except BaseException:
            child.kill(); child.wait()
            raise
        child.stdin.write(b'1')
        child.stdin.flush()
        child.stdin.close()
        return child

    def resume(self, pid):
        kernel = self.kernel
        kernel.CreateToolhelp32Snapshot.argtypes = [W.DWORD,W.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = W.HANDLE
        kernel.Thread32First.argtypes = [W.HANDLE,ctypes.POINTER(ThreadEntry)]
        kernel.Thread32Next.argtypes = [W.HANDLE,ctypes.POINTER(ThreadEntry)]
        kernel.OpenThread.argtypes = [W.DWORD,W.BOOL,W.DWORD]
        kernel.OpenThread.restype = W.HANDLE
        kernel.ResumeThread.argtypes = [W.HANDLE]
        kernel.ResumeThread.restype = W.DWORD
        snapshot = kernel.CreateToolhelp32Snapshot(4,0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry(); entry.dwSize = ctypes.sizeof(entry)
            available = kernel.Thread32First(snapshot,ctypes.byref(entry))
            while available:
                if entry.th32OwnerProcessID == pid:
                    thread = kernel.OpenThread(2,False,entry.th32ThreadID)  # THREAD_SUSPEND_RESUME
                    if not thread: raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if kernel.ResumeThread(thread)==0xffffffff: raise ctypes.WinError(ctypes.get_last_error())
                    finally: kernel.CloseHandle(thread)
                    return
                available = kernel.Thread32Next(snapshot,ctypes.byref(entry))
            raise RuntimeError('Suspended primary thread not found; process was not resumed')
        finally: kernel.CloseHandle(snapshot)

    def close(self):
        if getattr(self, 'handle', None):
            self.kernel.TerminateJobObject(self.handle, 137)
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            for child in getattr(self, 'children', []):
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass

    def contains(self, pid):
        kernel = self.kernel
        kernel.OpenProcess.argtypes = [W.DWORD,W.BOOL,W.DWORD]
        kernel.OpenProcess.restype = W.HANDLE
        kernel.IsProcessInJob.argtypes = [W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)]
        process = kernel.OpenProcess(0x1000,False,pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not process: raise ctypes.WinError(ctypes.get_last_error())
        try:
            member = W.BOOL()
            if not kernel.IsProcessInJob(process,self.handle,ctypes.byref(member)):
                raise ctypes.WinError(ctypes.get_last_error())
            return bool(member.value)
        finally: kernel.CloseHandle(process)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
