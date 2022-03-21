set_1 = {(1, 3)}
print(set_1)
set_1.update([(2, 4)])
print(set_1)
set_1.update([(1, )])
print(set_1)
print(len(set_1))
for x in set_1:
    for y in set_1:
        print(set(x).intersection(set(y)))