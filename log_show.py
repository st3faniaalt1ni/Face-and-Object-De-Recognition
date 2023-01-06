# cat run_2022-07-12-14\:13\:25/*.log | grep loss_D | cut -d ':' -f 2 | cut -d ',' -f1 | sed s/' '//g > log.log

import numpy as np

import pandas
import matplotlib.pyplot as plt
import sys
pd = pandas.read_csv('log.log')

plt.plot(pd.iloc[:])
plt.show()

plt.savefig(sys.argv[1]+'.png')