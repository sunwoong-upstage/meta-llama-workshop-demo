#!/usr/bin/env python3
"""
Simple deployment script for Customer Support Agent
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Success")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"   ❌ Error: {result.stderr.strip()}")
        return False
    return True

def main():
    print("🚀 Customer Support Agent Deployment")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("fastapi_app.py").exists():
        print("❌ Please run this script from the customer-support-agent directory")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("uv sync", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Copy production environment
    if Path("env.production").exists():
        if not run_command("cp env.production .env", "Setting up production environment"):
            print("❌ Failed to copy environment file")
            sys.exit(1)
        print("⚠️  Don't forget to update .env with your actual API keys!")
    
    # Start the server
    print("\n🎯 Starting server...")
    print("   Access your chatbot at: http://your-server-ip:8000")
    print("   API docs at: http://your-server-ip:8000/docs")
    print("   Press Ctrl+C to stop")
    print("\n" + "=" * 50)
    
    try:
        subprocess.run([
            "uv", "run", "uvicorn", "fastapi_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--workers", "1"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()

