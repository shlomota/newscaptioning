import os
import sys
from time import sleep
tepochs = [64]*100
tlist = [i.startswith('tepochs') for i in sys.argv]
if True in tlist:
    tepochs = sys.argv[tlist.index(True)][len('tepochs'):].split("_")

periodic = False

if 'periodic' in sys.argv:
    periodic = True

while True:

    if "_" in sys.argv[1]:
        os.system('clear')
        for ind,i in enumerate(sys.argv[1].split("*")):
            a = f"cat {i}.err | grep ' 1/{tepochs[ind]}' | wc -l"
            a = os.system(a)
        
    else:
        a = f"cat train2_{sys.argv[1]}.err | grep ' 1/{tepochs[0]}' | wc -l"
        a = os.system(a)

    if not periodic:
        break

    sleep(300)

