import os
import sys

tepochs = 64
if len(sys.argv) > 2:
    tepochs = sys.argv[2]

a = f"cat train2_{sys.argv[1]}.err | grep ' 1/{tepochs}' | wc -l"

a = os.system(a)

