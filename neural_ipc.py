from __future__ import annotations
import os
from multiprocessing import shared_memory
from typing import Any, Dict, Optional
import numpy as np
SHM_NAME: str = 'bat_sim_live_activations_shm'
SHM_SIZE: int = 1024
FILE_FALLBACK_PATH: str = os.path.join(os.path.dirname(__file__), 'live_activations.npz')

class ActivationWriter:

    def __init__(self, name: str=SHM_NAME) -> None:
        self.name = name
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.arr: Optional[np.ndarray] = None
        self.seq: float = 0.0
        self._init_shm()

    def _init_shm(self) -> None:
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=SHM_SIZE)
        except FileExistsError:
            try:
                existing = shared_memory.SharedMemory(name=self.name)
                existing.close()
                existing.unlink()
            except Exception:
                pass
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=SHM_SIZE)
            except Exception:
                try:
                    self.shm = shared_memory.SharedMemory(name=self.name)
                except Exception:
                    self.shm = None
        if self.shm is not None:
            self.arr = np.ndarray((128,), dtype=np.float64, buffer=self.shm.buf)
            self.arr[:] = 0.0

    def write(self, activations: Dict[str, np.ndarray], ping_freq: float=10.0, speed: float=75.0, heading: float=0.0) -> None:
        self.seq += 1.0
        u = activations.get('inputs', np.zeros(5, dtype=np.float64))
        h1 = activations.get('h1', np.zeros(32, dtype=np.float64))
        h2 = activations.get('h2', np.zeros(32, dtype=np.float64))
        out = activations.get('outputs', np.zeros(2, dtype=np.float64))
        if self.arr is not None:
            try:
                self.arr[0] = self.seq
                self.arr[1] = float(ping_freq)
                self.arr[2] = float(speed)
                self.arr[3] = float(heading)
                self.arr[4:9] = u[:5]
                self.arr[9:41] = h1[:32]
                self.arr[41:73] = h2[:32]
                self.arr[73:75] = out[:2]
            except Exception:
                pass
        try:
            tmp_path = FILE_FALLBACK_PATH + '.tmp.npz'
            np.savez(tmp_path, seq=float(self.seq), inputs=u, h1=h1, h2=h2, outputs=out, ping_freq=float(ping_freq), speed=float(speed), heading=float(heading))
            os.replace(tmp_path, FILE_FALLBACK_PATH)
        except Exception:
            pass

    def close(self) -> None:
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None
            self.arr = None

class ActivationReader:

    def __init__(self, name: str=SHM_NAME) -> None:
        self.name = name
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.arr: Optional[np.ndarray] = None

    def connect(self) -> bool:
        if self.arr is not None:
            return True
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
            self.arr = np.ndarray((128,), dtype=np.float64, buffer=self.shm.buf)
            return True
        except Exception:
            self.shm = None
            self.arr = None
            return False

    def read(self) -> Optional[Dict[str, Any]]:
        if self.connect() and self.arr is not None:
            try:
                seq = float(self.arr[0])
                if seq > 0:
                    return {'seq': seq, 'inputs': self.arr[4:9].copy(), 'h1': self.arr[9:41].copy(), 'h2': self.arr[41:73].copy(), 'outputs': self.arr[73:75].copy(), 'ping_freq': float(self.arr[1]), 'speed': float(self.arr[2]), 'heading': float(self.arr[3])}
            except Exception:
                pass
        if os.path.exists(FILE_FALLBACK_PATH):
            try:
                with np.load(FILE_FALLBACK_PATH) as data:
                    return {'seq': float(data.get('seq', 0.0)), 'inputs': data['inputs'], 'h1': data['h1'], 'h2': data['h2'], 'outputs': data['outputs'], 'ping_freq': float(data.get('ping_freq', 10.0)), 'speed': float(data.get('speed', 75.0)), 'heading': float(data.get('heading', 0.0))}
            except Exception:
                pass
        return None

    def close(self) -> None:
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            self.shm = None
            self.arr = None
