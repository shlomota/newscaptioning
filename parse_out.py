import os
import sys
import numpy as np

TAT_FOLDER = "/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/newscaptioning/"
OUT_FILE = "out_prio.out"

if len(sys.argv) > 1:
    OUT_FILE = sys.argv[1]

def main():

    out = open(os.path.join(TAT_FOLDER, OUT_FILE), 'r').read()
    out = out[out.find("0\nimage_id:"):]
    p = out.strip().split("\n<N>\n")
    p = [dict([i.split(": ")  for i in j.split("\n") if ":" in i]) for j in p]
    p.sort(key=lambda x: (x['image_id'], x['gentype']))
    #print("\n".join([f"{j}: {i[j]}" for i in p for j in i]))

    pairs = [p[i*2:i*2+2] for i in range(int(len(p)/2))]
    c = [p[0]['image_id'] == p[1]['image_id'] for p in pairs]
    if False in c:
        print('error in image_id in pairs')
        return

    c = [(p[0]['gentype'] == '1' and p[1]['gentype'] == '2') for p in pairs]
    if False in c:
        print('error in gentype in pairs')
        return

    scores = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']
    pairs = [dict([(s, float(p[1][s]) - float(p[0][s])) for s in scores] + [('image_id', p[0]['image_id'])]) for p in pairs]
    print(f'articles:{len(pairs)}')

    open(os.path.join(TAT_FOLDER, "pocpairs.out"), 'w').write(str(pairs))
    print()
    print()

    for s in scores:
        print(s)
        s = np.array([i[s] for i in pairs])
        print(f'mean: {s.mean()}')
        print(f'improve: {(s>0).sum()}')
        print(f'worse: {(s<0).sum()}')
        print(f'same: {(s==0).sum()}')
        print()

if __name__ == '__main__':
    main()