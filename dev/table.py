import simulacra as si

headers = ("a", "b", "c")
rows = ((1, 2, 3), None, (3, 4, 5, 6))

print(si.utils.table(headers, rows))
