import subprocess
from pathlib import Path
import sys

data_dir = "src/output_dl_seed99999"
scripts = [
    "src/timeseries_semester_transformer.py",
    "src/timeseries_exam_transformer.py",
    "src/timeseries_semester.py", 
    "src/timeseries_exam.py"
]

print("Starte gradeblind Training im Hintergrund...")
for script in scripts:
    print(f"Starte {script} mit mode=gradeblind...")
    subprocess.run([sys.executable, script, "--data_dir", data_dir, "--mode", "gradeblind"], check=True)
print("Gradeblind Training abgeschlossen!")
