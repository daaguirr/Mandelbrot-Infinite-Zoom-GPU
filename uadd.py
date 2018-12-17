def uadd(a, b, ans):
    n = len(a)
    carry = 0
    for i in reversed(range(n)):
        abc = a[i] + b[i] + carry
        ans[i] = abc
        carry = abc >> 32
