import matplotlib.pyplot as plt
import numpy as np

# 1. Define files and directories
cpuResults = "CML/results/cpu_results.txt"
gpuResults = "CML/results/gpu_results.txt"
plotsDir = "CML/results"


# 2. Initialize the matrices with the results
cpuCML = np.loadtxt(open(cpuResults, "rb"), delimiter=",", skiprows=2)
gpuCML = np.loadtxt(open(gpuResults, "rb"), delimiter=",", skiprows=2)


# 3. Compute the plots and store them
# CPU
plt.imshow(cpuCML, cmap="viridis", aspect="auto")
plt.colorbar(label='Cell Value')
plt.title(f'CML Evolution for CPU')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')
plt.savefig(plotsDir + "/cpu_results.png")
plt.close()

# GPU
plt.imshow(gpuCML, cmap="viridis", aspect="auto")
plt.colorbar(label='Cell Value')
plt.title(f'CML Evolution for GPU')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')
plt.savefig(plotsDir + "/gpu_results.png")
plt.close()
