import sys
import time
from pathlib import Path
import os

username = "shlomotannor"
if len(sys.argv)>1:
    username = sys.argv[1]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



while True:
    if os.popen(f'squeue -u {username}').read().find(username[:5]) == -1:
        print('                                                  ', end='\r')
        print(bcolors.OKGREEN+"Done."+bcolors.ENDC, end='\r')
    else:
        print('                                                  ', end='\r')
        print(bcolors.WARNING+"Running..."+bcolors.ENDC, end='\r')

    time.sleep(5)
