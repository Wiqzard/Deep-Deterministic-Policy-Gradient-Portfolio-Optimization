from tqdm import tqdm


for i in tqdm(range(20000000), total=20000000, position=0, leave=True, bar_format='l_bar'):
    i = 2*i - i
    i += 1
    i = i/ i**2 