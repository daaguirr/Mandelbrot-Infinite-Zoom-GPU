import numpy as np
from uadd import uadd


def umul(a, b, ans):
    n = len(a)
    aux = a.copy()
    tmp = np.zeros_like(a)
    tmp2 = np.zeros_like(a)
    for i in reversed(range(n)):
        umuli(a, b[i], tmp)
        uadd(ans, tmp, tmp2)
        ans = tmp2.copy()
        lsh32(aux, 1)
