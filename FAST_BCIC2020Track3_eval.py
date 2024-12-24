import numpy as np

accuracy = []
for FOLD in range(15):
    logf = f"Results/FAST/{FOLD}-Tune.csv"
    data = np.loadtxt(logf, delimiter=',', dtype=int)
    pred, label = data[:, 0], data[:, 1]
    accuracy.append(np.mean(pred == label))

print(f"Accuracy: {np.mean(accuracy):4f}, Std: {np.std(accuracy):4f}")
