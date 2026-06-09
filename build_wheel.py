import sys
import subprocess


subprocess.check_call([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'])