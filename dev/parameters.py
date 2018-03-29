import simulacra as si

p = si.cluster.Parameter('hello', value = 'foo')
print(p)

e = si.cluster.Parameter('hello', value = range(1004), expandable = True)
print(e)
