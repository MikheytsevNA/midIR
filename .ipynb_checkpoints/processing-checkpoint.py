import numpy as np
import matplotlib.pyplot as plt

list = []
with open("data.txt", 'r') as data:
    for line in data:
        list.append(line.split("    "))
thin_4 = []
thin_7 = []
needle = []
super = []
n = []
for i in list:
    if i[0] == 'thin' and i[1] == '4e-5cm':
        thin_4.append(float(i[3][:-2]))
        n.append(float(i[2]))
    elif i[0] == 'thin' and i[1] == '7e-5cm':
        thin_7.append(float(i[3][:-2]))
    elif i[0] == 'super':
        super.append(float(i[2][:-2]))
    else:
        needle.append(float(i[3][:-2]))
"""print(n)
print(thin_7)"""
thin_7 = [thin_7 for _,thin_7 in sorted(zip(n,thin_7))]
thin_4 = [thin_4 for _,thin_4 in sorted(zip(n,thin_4))]
super = [super for _,super in sorted(zip(n,super))]
needle = [needle for _,needle in sorted(zip(n,needle))]
n.sort()
"""print(sorted(n))
print(thin_7)"""
fig = plt.figure(figsize=(10,10))
plt.plot(n,thin_7 , label = r'thin 7e-5cm')
plt.plot(n,thin_4, label = r'thin 4e-5cm')
plt.plot(n,super, label = r'super gauss')
plt.plot(n,needle, label = r'needle')
plt.legend(loc = 'best', prop={'size': 20})
plt.xlabel(r"$n_0$, $n_{cr}$", fontsize = 26)
plt.ylabel(r"$\frac{W_i}{W_{THz}}$, %", fontsize = 26)
plt.xticks(np.arange(3,61,3))
plt.yticks(np.arange(0.0,0.9,0.1))
plt.grid()
fig.savefig("./final.png", dpi=fig.dpi)
plt.show()
