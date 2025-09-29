'''import matplotlib.pyplot as plt
data = []
with open("data_blood.txt", "r") as file:
    data = file.readlines()
value = {}
for point in data:
    value.update({float(point.split()[0]):float(point.split()[1])})
print(*value)

vst = [z*5/(2**12) for z in value.values()]
Ist = [v/0.5 for v in vst]

plt.figure()
plt.plot(value.keys(),value.values())

plt.figure()
plt.plot(value.keys(),Ist)

plt.show()
'''
