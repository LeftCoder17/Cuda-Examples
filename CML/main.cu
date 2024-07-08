#include "cmlCPU.h"
#include "cmlGPU.h"
#include <ctime>
#include <iostream>

int main()
{   
    int nNodesList[4] = {100, 1000, 5000};
    int nStepsList[4] = {100, 1000, 2000, 3000};
    float epsilon = 0.2;
    float mu = 3.06;

    for (int nNodes : nNodesList){
        for (int nSteps : nStepsList)
        {
            // CPU
            clock_t cpuClock = clock();
            cmlCPU cml = cmlCPU(nNodes, nSteps, epsilon, mu);
            cml.init_lattice(0);
            cml.compute_lattice();
            cml.export_evolution("CML/results/cpu_results.txt");
            cpuClock = clock() - cpuClock;

            // GPU
            clock_t gpuClock = clock();
            cmlGPU cml2 = cmlGPU(nNodes, nSteps, epsilon, mu);
            cml2.init_lattice(0);
            cml2.compute_lattice();
            cml2.export_evolution("CML/results/gpu_results.txt");
            cudaDeviceSynchronize();
            gpuClock = clock() - gpuClock;


            // Comparation
            double cpuTime = (double) cpuClock / CLOCKS_PER_SEC;
            double gpuTime = (double) gpuClock / CLOCKS_PER_SEC;
            double totalTime = cpuTime + gpuTime;
            std::cout << "For nNodes = " << nNodes << " and nSteps = " << nSteps << std::endl;
            std::cout << "CPU took " << cpuTime << " seconds (" << cpuTime*100/totalTime << "%)" << std::endl;
            std::cout << "GPU took " << gpuTime << " seconds (" << gpuTime*100/totalTime << "%)" << std::endl;
        }
    }

    return 0;
}