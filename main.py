"""
main.py
=======
Entry point for the AMR Multi-Agent Framework.

Usage:
  # Run with dashboard (recommended):
  streamlit run dashboard/app.py

  # Run headless (terminal only, no UI):
  python main.py

  # Run headless with custom config:
  python main.py --amrs 3 --speed 0.5

Author: AMR Multi-Agent Framework
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/amr_system.log"),
        logging.StreamHandler(),
    ]
)

from core.orchestrator import Orchestrator


async def run(config: dict):
    orch = Orchestrator(config)

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orch.stop()))

    await orch.run_forever()


def main():
    parser = argparse.ArgumentParser(description="AMR Multi-Agent Framework")
    parser.add_argument("--amrs",  type=int,   default=5,   help="Number of AMRs (default: 5)")
    parser.add_argument("--grid",  type=int,   default=20,  help="Grid size (default: 20)")
    parser.add_argument("--speed", type=float, default=1.0, help="Tick rate in seconds (default: 1.0)")
    args = parser.parse_args()

    config = {
        "num_amrs":                  args.amrs,
        "grid_size":                 args.grid,
        "tick_rate":                 args.speed,
        "energy_low_threshold":      20.0,
        "energy_critical_threshold": 10.0,
        "anomaly_sensitivity":       0.05,
        "task_work_duration":        3.0,
        "db_path":                   "data/amr_history.db",
    }

    print("\n" + "=" * 60)
    print("  AMR Multi-Agent Framework")
    print("  Private 5G Fleet Management System")
    print("=" * 60)
    print(f"  AMRs: {config['num_amrs']} | Grid: {config['grid_size']}x{config['grid_size']}")
    print(f"  Tick rate: {config['tick_rate']}s | Dashboard: streamlit run dashboard/app.py")
    print("=" * 60 + "\n")

    asyncio.run(run(config))


if __name__ == "__main__":
    main()