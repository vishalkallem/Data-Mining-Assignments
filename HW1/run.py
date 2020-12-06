import os
from time import time

if __name__ == '__main__':
    start = time()
    os.system('python vishal_kallem_task1.py Gamergate.json output1.json')
    os.system('python vishal_kallem_task2.py Gamergate.json output2.json')
    os.system('python vishal_kallem_task3.py tweets output3.json')
    print('Elapsed Time is: ', time()-start)
