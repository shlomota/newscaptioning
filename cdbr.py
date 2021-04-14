from glob import glob

print(f'train: {len(glob("dbr/*[!m]"))-2}, test: {len(glob("dbr/test/*[!m]"))-1}')

