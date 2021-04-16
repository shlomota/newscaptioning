from glob import glob
import time
from getmissingid import main as updateIds
from os import system

i = 0

while True:
    print(f'{time.ctime()}: train: {len(glob("dbr/*[!m]"))-4}')  #, test: {len(glob("dbr/test/*[!m]"))-1}')
    time.sleep(120)

    i += 1
    if i == 10:
        system('clear')
        i = 0
        new_len = updateIds(['update'])
        print(f"updated ids to {new_len}")