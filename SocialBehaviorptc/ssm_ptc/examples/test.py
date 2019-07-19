from tqdm import tqdm
import time
import sys


values = range(3)
with tqdm(total=len(values), file=sys.stdout) as pbar:
    for i in values:
        pbar.set_description('processed: %d' % (1 + i))
        pbar.update(1)
        time.sleep(1)