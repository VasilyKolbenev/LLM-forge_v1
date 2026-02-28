"""Compute target management for remote GPU resources."""

import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_forge.compute.ssh import SSHConnection

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("./data/compute_targets.json")


@dataclass
class ComputeTarget:
    """A remote or local compute target."""

    id: str
    name: str
    host: str
    user: str
    port: int = 22
    key_path: str | None = None
    gpu_count: int = 0
    gpu_type: str = ""
    vram_gb: float = 0
    status: str = "unknown"
    added_at: str = ""
    last_seen: str | None = None


@dataclass
class ConnectionTestResult:
    """Result of testing SSH connection to a target."""

    success: bool
    message: str
    latency_ms: float = 0
    gpu_info: str = ""


class ComputeManager:
    """Manages compute targets (local + remote GPU machines).

    Args:
        store_path: Path to JSON file storing targets.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path or DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._save([])

    def add_target(
        self,
        name: str,
        host: str,
        user: str,
        port: int = 22,
        key_path: str | None = None,
    ) -> ComputeTarget:
        """Add a new compute target.

        Args:
            name: Display name for the target.
            host: Hostname or IP address.
            user: SSH username.
            port: SSH port.
            key_path: Path to SSH private key.

        Returns:
            Created ComputeTarget.
        """
        target = ComputeTarget(
            id=str(uuid.uuid4())[:8],
            name=name,
            host=host,
            user=user,
            port=port,
            key_path=key_path,
            added_at=datetime.now().isoformat(),
        )
        targets = self._load()
        targets.append(asdict(target))
        self._save(targets)
        logger.info("Added compute target: %s (%s@%s)", name, user, host)
        return target

    def remove_target(self, target_id: str) -> bool:
        """Remove a compute target.

        Args:
            target_id: Target ID to remove.

        Returns:
            True if removed, False if not found.
        """
        targets = self._load()
        original = len(targets)
        targets = [t for t in targets if t["id"] != target_id]
        if len(targets) < original:
            self._save(targets)
            return True
        return False

    def get_target(self, target_id: str) -> ComputeTarget | None:
        """Get a single target by ID.

        Args:
            target_id: Target ID.

        Returns:
            ComputeTarget or None.
        """
        for t in self._load():
            if t["id"] == target_id:
                return ComputeTarget(**{
                    k: v for k, v in t.items()
                    if k in ComputeTarget.__dataclass_fields__
                })
        return None

    def list_targets(self) -> list[ComputeTarget]:
        """List all compute targets.

        Returns:
            List of ComputeTarget instances.
        """
        return [
            ComputeTarget(**{
                k: v for k, v in t.items()
                if k in ComputeTarget.__dataclass_fields__
            })
            for t in self._load()
        ]

    def test_connection(self, target_id: str) -> ConnectionTestResult:
        """Test SSH connection to a target.

        Args:
            target_id: Target to test.

        Returns:
            ConnectionTestResult with success/failure info.
        """
        target = self.get_target(target_id)
        if not target:
            return ConnectionTestResult(
                success=False, message="Target not found"
            )

        import time

        start = time.monotonic()
        try:
            conn = SSHConnection(
                host=target.host,
                user=target.user,
                port=target.port,
                key_path=target.key_path,
            )
            conn.connect()
            stdout, stderr, code = conn.exec_command("echo ok", timeout=10)
            latency = (time.monotonic() - start) * 1000
            conn.close()

            if code != 0:
                return ConnectionTestResult(
                    success=False,
                    message=f"Command failed: {stderr.strip()}",
                    latency_ms=latency,
                )

            # Update status
            self._update_target(target_id, status="online", last_seen=datetime.now().isoformat())

            return ConnectionTestResult(
                success=True,
                message="Connected successfully",
                latency_ms=round(latency, 1),
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            self._update_target(target_id, status="offline")
            return ConnectionTestResult(
                success=False,
                message=str(e),
                latency_ms=round(latency, 1),
            )

    def detect_remote_hardware(self, target_id: str) -> dict[str, Any]:
        """Detect hardware on a remote target.

        Args:
            target_id: Target to probe.

        Returns:
            Dict with gpu_count, gpu_type, vram_gb.
        """
        target = self.get_target(target_id)
        if not target:
            return {"error": "Target not found"}

        try:
            conn = SSHConnection(
                host=target.host,
                user=target.user,
                port=target.port,
                key_path=target.key_path,
            )
            conn.connect()

            # Try nvidia-smi
            stdout, _, code = conn.exec_command(
                "nvidia-smi --query-gpu=count,name,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null || echo 'no-gpu'",
                timeout=15,
            )
            conn.close()

            if "no-gpu" in stdout or code != 0:
                return {"gpu_count": 0, "gpu_type": "CPU only", "vram_gb": 0}

            lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
            if not lines:
                return {"gpu_count": 0, "gpu_type": "CPU only", "vram_gb": 0}

            parts = lines[0].split(",")
            gpu_count = len(lines)
            gpu_type = parts[1].strip() if len(parts) > 1 else "Unknown"
            vram_mb = float(parts[2].strip()) if len(parts) > 2 else 0
            vram_gb = round(vram_mb / 1024, 1)

            # Update target with detected info
            self._update_target(
                target_id,
                gpu_count=gpu_count,
                gpu_type=gpu_type,
                vram_gb=vram_gb,
                status="online",
                last_seen=datetime.now().isoformat(),
            )

            return {
                "gpu_count": gpu_count,
                "gpu_type": gpu_type,
                "vram_gb": vram_gb,
            }
        except Exception as e:
            return {"error": str(e)}

    def _update_target(self, target_id: str, **kwargs: Any) -> None:
        """Update fields on a target."""
        targets = self._load()
        for t in targets:
            if t["id"] == target_id:
                t.update(kwargs)
                break
        self._save(targets)

    def _load(self) -> list[dict]:
        with open(self.store_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, targets: list[dict]) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(targets, f, ensure_ascii=False, indent=2)
