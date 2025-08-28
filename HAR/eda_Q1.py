import numpy as np
import matplotlib.pyplot as plt

# training data
X = np.load("training_data.npy")
y = np.load("training_labels.npy")

# ACTIVITIES
ACTIVITIES = { 1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs", 4: "Sitting", 5: "Standing", 6: "Laying" }

# Pick one sample from each activity
samples = {}
for label in np.unique(y):
    # first index in y where y == label
    idx = np.where(y == label)[0][0]   # np.where(y == label)[0] gives array of indexes where y == label
    samples[label] = X[idx] # adding whole sample of (500,3) size

# Plot signals

for i, label in enumerate(samples.keys()):
    plt.subplot(3, 2, i+1)
    plt.plot(samples[label][:,0], color="r", label="accx")
    plt.plot(samples[label][:,1], color="g", label="accy")
    plt.plot(samples[label][:,2], color="b", label="accz")
    plt.title(ACTIVITIES[label])
    plt.xlabel("Time (samples)")
    plt.ylabel("Acceleration (g)")
    plt.legend(fontsize=6)

plt.tight_layout()
plt.show()
