#!/usr/bin/env python3
"""
Update ~/.ssh/config with the current RunPod spot pod's IP and SSH port.

Run this on your LOCAL machine (where Cursor runs). You do NOT need to be
connected to the pod. You DO need runpodctl installed and configured with
your RunPod API key (e.g. runpodctl config --apiKey=...) so it can fetch
pod details. After the script runs, connect with: ssh runpod-spot

Run after a pod restarts (e.g. spot preemption) to refresh the IP/port.
"""

import os
import subprocess
import json
import re
import argparse
from typing import Optional

# --- CONFIGURATION ---
HOST_ALIAS = "runpod-spot"  # This is the name you use in Cursor
SSH_CONFIG_PATH = os.path.expanduser("~/.ssh/config")
IDENTITY_FILE = os.path.expanduser("~/.ssh/id_ed25519")

# Private/internal IP ranges (we must use public IP for SSH from outside)
_PRIVATE_IP_PATTERN = re.compile(
    r"^"
    r"(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # 10.0.0.0/8
    r"|(?:100\.(?:6[4-9]|[7-9][0-9]|1[0-2][0-9])\.\d{1,3}\.\d{1,3})"  # 100.64.0.0/10
    r"|(?:172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})"  # 172.16.0.0/12
    r"|(?:192\.168\.\d{1,3}\.\d{1,3})"  # 192.168.0.0/16
    r"$"
)


def _is_private_ip(ip: str) -> bool:
    """Return True if the IP is private/internal (not routable from the internet)."""
    return bool(ip and _PRIVATE_IP_PATTERN.match(ip.strip()))


def _pick_public_ip(*candidates: str) -> Optional[str]:
    """Return the first non-private IP, or the first candidate if all private."""
    for ip in candidates:
        if not ip or not isinstance(ip, str):
            continue
        ip = ip.strip()
        if not _is_private_ip(ip):
            return ip
    return next((c.strip() for c in candidates if c and isinstance(c, str)), None)


def _find_ssh_public_port(ports: list) -> Optional[int]:
    """From a list of port mappings, return the public/host port that maps to container port 22."""
    for p in ports:
        if not isinstance(p, (dict, list, tuple)):
            continue
        if isinstance(p, dict):
            private_port = p.get("privatePort") or p.get("containerPort")
            public_port = p.get("publicPort") or p.get("hostPort")
            try:
                if (private_port == 22 or (private_port is not None and int(private_port) == 22)) and public_port is not None:
                    return int(public_port)
            except (TypeError, ValueError):
                pass
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                a, b = p[0], p[1]
                ai = int(a) if a is not None else None
                bi = int(b) if b is not None else None
            except (TypeError, ValueError):
                continue
            if ai == 22 and bi is not None:
                return int(bi)
            if bi == 22 and ai is not None:
                return int(ai)
    return None


def _parse_pod_list_json(stdout: str):
    """Parse output of 'runpodctl pod list' (new CLI, JSON by default)."""
    data = json.loads(stdout)
    pods = data if isinstance(data, list) else ([data] if data else [])
    for pod in pods:
        # New CLI / API: runtime status may be under 'runtime' or top-level
        runtime = pod.get("runtime") or pod
        status = (
            runtime.get("status")
            or runtime.get("runtimeStatus")
            or pod.get("desiredStatus")
        )
        if status != "RUNNING" and pod.get("runtimeStatus") != "Running":
            continue
        # Prefer public IP: collect all IP-like fields and pick first public one
        machine = pod.get("machine") or {}
        ip = _pick_public_ip(
            pod.get("publicIp"),
            pod.get("publicIP"),
            pod.get("ip"),
            machine.get("podHostId"),
            machine.get("publicIp"),
        )
        # Ports: find the mapping where container/private port is 22, use public port
        ports = (
            pod.get("exposePorts")
            or pod.get("ports")
            or (runtime.get("ports") if isinstance(runtime, dict) else None)
            or []
        )
        if not isinstance(ports, list):
            ports = []
        ssh_port = _find_ssh_public_port(ports)
        if ip and ssh_port is not None:
            return ip, int(ssh_port)
    return None, None


def _parse_get_pod_output(stdout: str):
    """Parse output of legacy 'runpodctl get pod --allfields' (no --format)."""
    # Some versions still output JSON without --format
    stripped = stdout.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return _parse_pod_list_json(stdout)
        except (json.JSONDecodeError, TypeError):
            pass

    # RunPod table PORTS column: "213.173.98.94:12157->22 (pub,tcp),100.65.x:60819->8888 (prv,http)..."
    # Use the public SSH mapping (pub,tcp) that maps to container port 22.
    ssh_pub_match = re.findall(
        r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{2,5})->22\s*\(pub,tcp\)",
        stdout,
    )
    for ip, port_str in ssh_pub_match:
        if not _is_private_ip(ip):
            return ip, int(port_str)

    # Fallback: find IP and port with other patterns
    ip_re = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\b"
    )
    all_ips = ip_re.findall(stdout)
    ip = _pick_public_ip(*all_ips) if all_ips else None

    # Find public port that maps to SSH (container port 22)
    ssh_public_port = None
    mapping = re.search(r"(\d{3,5})\s*[->:]\s*22\b", stdout)
    if mapping:
        ssh_public_port = int(mapping.group(1))
    if ssh_public_port is None:
        mapping2 = re.search(r"\b22\s*[->:]\s*(\d{3,5})\b", stdout)
        if mapping2:
            ssh_public_port = int(mapping2.group(1))
    if ssh_public_port is None:
        for m in re.finditer(r"(?:privatePort|containerPort|private_port)[\s:]+22\b", stdout, re.I):
            snippet = stdout[max(0, m.start() - 20) : m.end() + 80]
            pub = re.search(r"(?:publicPort|hostPort|public_port)[\s:]+(\d{3,5})", snippet, re.I)
            if pub:
                ssh_public_port = int(pub.group(1))
                break
    if ssh_public_port is None:
        for m in re.finditer(r"(?:publicPort|hostPort|public_port)[\s:]+(\d{3,5})", stdout, re.I):
            snippet = stdout[m.start() : m.end() + 100]
            if re.search(r"(?:privatePort|containerPort|private_port)[\s:]+22\b", snippet, re.I):
                ssh_public_port = int(m.group(1))
                break

    if ip and ssh_public_port is not None:
        return ip, ssh_public_port
    return None, None


def get_pod_info(debug: bool = False):
    """Get IP and SSH port of the first running RunPod pod via runpodctl."""
    # Try legacy CLI first (runpodctl get pod --allfields) — many installs don't have "pod list"
    runpodctl_not_found = False
    try:
        result_legacy = subprocess.run(
            ["runpodctl", "get", "pod", "--allfields"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        runpodctl_not_found = True
        result_legacy = None
    except subprocess.TimeoutExpired:
        result_legacy = None

    if result_legacy and result_legacy.returncode == 0 and result_legacy.stdout.strip():
        ip, port = _parse_get_pod_output(result_legacy.stdout)
        if ip and port is not None:
            return ip, port
        if debug:
            print("--- runpodctl get pod --allfields (stdout) ---")
            print(result_legacy.stdout)
            print("--- end ---")

    # New CLI: runpodctl pod list (JSON by default)
    try:
        result = subprocess.run(
            ["runpodctl", "pod", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        runpodctl_not_found = True
        result = None
    except subprocess.TimeoutExpired:
        result = None

    if result and result.returncode == 0 and result.stdout.strip():
        try:
            ip, port = _parse_pod_list_json(result.stdout)
            if ip and port is not None:
                return ip, port
        except json.JSONDecodeError:
            pass

    # No pod info from either command
    if runpodctl_not_found:
        print("❌ 'runpodctl' not found. Install RunPod CLI and log in.")
    elif result_legacy and result_legacy.returncode == 0:
        print("❌ Could not parse pod IP/port from runpodctl output. Run with --debug to see the raw output.")
    elif result_legacy and result_legacy.returncode != 0:
        print(f"❌ runpodctl failed: {result_legacy.stderr or result_legacy.stdout}")
    elif result and result.returncode != 0:
        print(f"❌ runpodctl failed: {result.stderr or result.stdout}")
    return None, None


def update_config(ip, port):
    """Write or replace the Host entry for HOST_ALIAS in ~/.ssh/config."""
    new_entry = (
        f"Host {HOST_ALIAS}\n"
        f"    HostName {ip}\n"
        f"    User root\n"
        f"    Port {port}\n"
        f"    IdentityFile {IDENTITY_FILE}\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
    )

    try:
        content = ""
        if os.path.exists(SSH_CONFIG_PATH):
            with open(SSH_CONFIG_PATH, "r") as f:
                content = f.read()
    except OSError as e:
        print(f"❌ Cannot read SSH config: {e}")
        return

    # Match existing block: "Host runpod-spot" plus any following indented lines
    pattern = rf"Host {re.escape(HOST_ALIAS)}\n(?:[ \t].*\n?)*"
    if re.search(pattern, content):
        content = re.sub(pattern, new_entry, content)
    else:
        content = content.rstrip() + "\n\n" + new_entry

    try:
        os.makedirs(os.path.dirname(SSH_CONFIG_PATH), mode=0o700, exist_ok=True)
        with open(SSH_CONFIG_PATH, "w") as f:
            f.write(content)
    except OSError as e:
        print(f"❌ Cannot write SSH config: {e}")
        return

    print(f"✅ Success! Cursor can now connect to {ip}:{port} using alias '{HOST_ALIAS}'")


def main():
    parser = argparse.ArgumentParser(
        description="Update ~/.ssh/config with your RunPod pod's IP and SSH port (run locally, no SSH to pod needed).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw runpodctl output when parsing fails (to adapt the script to your CLI format).",
    )
    args = parser.parse_args()

    ip, port = get_pod_info(debug=args.debug)
    if ip and port is not None:
        update_config(ip, port)
    else:
        print("❌ No running pods found. Start your pod on RunPod first.")


if __name__ == "__main__":
    main()
