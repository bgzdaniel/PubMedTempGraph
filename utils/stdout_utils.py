import os
import sys

class StdoutSwitch():
    def __init__(self):
        self.original_stdout = sys.stdout

    def off(self):
        f = open(os.devnull, 'w')
        sys.stdout = f

    def on(self):
        sys.stdout = self.original_stdout