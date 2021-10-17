import numpy as np



a = np.array([
[53.28,54.39,54.31],
[46.28,47.49,45.38],
[66.55,68.98,68.30],
[70.81,68.67,68.79 ]
])
sum_ = 0.
for i in range(a.shape[0]):
    mean = np.mean(a[i])
    std = np.std(a[i])
    print(i,mean,std)
    sum_ += mean

sum_ /= a.shape[0]
print("Avg:", sum_)