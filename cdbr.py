from glob import glob
import time

while True:
    print(f'{time.ctime()}: train: {len(glob("dbr/*[!m]"))-4}, test: {len(glob("dbr/test/*[!m]"))-1}')
    time.sleep(120)

