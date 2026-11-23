#!/usr/bin/env python3
"""
Launcher for Stock Market Prediction Academic Showcase Dashboard.

Run with:
    python run_dashboard.py
or:
    python run_dashboard.py --port 8050
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
from dashboard.server import run_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Stock Market Forecasting Presentation Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server (default: 8050)")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    run_server(port=args.port, bind=args.bind)
