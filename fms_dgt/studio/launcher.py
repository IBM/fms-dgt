# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import os
import subprocess
import sys
import time
import urllib.request

# Third Party
from dotenv import load_dotenv
import psutil

STUDIO_PORT = int(os.environ.get("DIGIT_STUDIO_PORT", 4720))
_STUDIO_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "studio", "dist")


def _studio_dist_path() -> str:
    return os.path.abspath(_STUDIO_DIST)


def is_display_available() -> bool:
    """Return False on headless Linux (no DISPLAY or WAYLAND_DISPLAY set)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def is_studio_running(port: int = STUDIO_PORT) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=1)
        return True
    except Exception:
        return False


def stop_studio(port: int = STUDIO_PORT) -> None:
    """Terminate the Studio process listening on the given port."""
    targets = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    targets.append(proc)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not targets:
        print(f"DiGiT Studio: nothing running on port {port}")
        return

    for proc in targets:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            proc.kill()

    print(f"DiGiT Studio: stopped (port {port})")


def main() -> None:
    """Entry point for the `digit-studio` console script."""

    load_dotenv()

    parser = argparse.ArgumentParser(description="DiGiT Studio run monitor.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "stop"],
        default="start",
        help="'start' launches Studio (default), 'stop' terminates a running instance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("DGT_OUTPUT_DIR"),
        help="Path to the DiGiT output directory to monitor (default: $DGT_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=STUDIO_PORT,
        help=f"Port DiGiT Studio runs on (default: {STUDIO_PORT}).",
    )
    args = parser.parse_args()

    if args.action == "stop":
        stop_studio(port=args.port)
    else:
        if not args.output_dir:
            print(
                "DiGiT Studio: --output-dir is required (or set $DGT_OUTPUT_DIR).\n"
                "Example: digit-studio start --output-dir ./output"
            )
            sys.exit(1)
        launch_studio(output_dir=args.output_dir, port=args.port)


def launch_studio(output_dir: str, port: int = STUDIO_PORT) -> None:
    """Launch DiGiT Studio if a display is available and it is not already running."""
    if not is_display_available():
        print("DiGiT Studio: skipped (no display detected — headless environment)")
        return

    if is_studio_running(port):
        print(f"DiGiT Studio is already running at http://localhost:{port}")
        return

    node_server = os.path.join(_studio_dist_path(), "server.js")
    if not os.path.exists(node_server):
        print(
            "DiGiT Studio: dist not found. "
            "Run: cd studio && npm install && npm run build && "
            "cp -r .next/standalone ../studio/dist && cp -r .next/static ../studio/dist/.next/static"
        )
        return

    env = {
        **os.environ,
        "PORT": str(port),
        "DGT_OUTPUT_DIR": os.path.abspath(output_dir),
    }

    subprocess.Popen(
        ["node", node_server],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Poll until ready (up to 10 seconds)
    for _ in range(20):
        if is_studio_running(port):
            print(f"DiGiT Studio is running at http://localhost:{port}")
            return
        time.sleep(0.5)

    print(
        "DiGiT Studio: server did not respond within 10s, check that Node.js is installed and on your PATH"
    )
