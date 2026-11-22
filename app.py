"""
PetVision AI - Root Entrypoint
==============================
Main launcher for Hugging Face Spaces and Streamlit Community Cloud.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

app_script = os.path.join(app_dir, "app.py")
with open(app_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, app_script, "exec"), globals(), locals())
