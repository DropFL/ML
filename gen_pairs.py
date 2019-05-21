import itertools
import sys

n = int(sys.argv[1])
m = int(sys.argv[2])

for i in itertools.product(range(n), range(m)):
    print(i[0] + 1, i[1] + 1)
