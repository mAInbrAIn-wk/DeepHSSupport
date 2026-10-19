import sys
from pathlib import Path

# Add src and legacy_code to path
sys.path.insert(0, str(Path("src").absolute()))
sys.path.insert(0, str(Path("legacy_code/archive_scripts").absolute()))

from deepsupport.runners.run_hypothesis_investigation import main

if __name__ == '__main__':
    main()
