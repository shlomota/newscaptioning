from glob import glob

print(f'train: {len(glob("dbr/*"))-2}, test: {len(glob("dbr/test/*"))-1}')

