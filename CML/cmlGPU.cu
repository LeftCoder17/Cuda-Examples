#include "cmlGPU.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>


#define BLOCK_SIZE 32


__device__
double logistic_map_cu(float mu, float x)
{
    return mu * x * (1- x);
}


__device__
float coupledmaplattice_cu(float epsilon, float mu, float x, float xLeft, float xRight)
{
    return (1 - epsilon) * logistic_map_cu(mu, x) + (epsilon / 2) * (logistic_map_cu(mu, xRight) + logistic_map_cu(mu, xLeft));
}

__global__
void computeCML(float *lattices, int nNodes, int nSteps, float epsilon, float mu)
{   
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nNodes) return;

    for (int step = 1; step <= nSteps; step++)
    {
        __syncthreads();
        
        float xLeft, xRight;
        // Apply Periodic Boundary Conditions
        xLeft = lattices[(step - 1) * nNodes + (node == 0 ? nNodes - 1 : node - 1)];
        xRight = lattices[(step - 1) * nNodes + (node == nNodes - 1 ? 0 : node + 1)];

        lattices[step * nNodes + node] = coupledmaplattice_cu(epsilon, mu,
                                                             lattices[(step - 1) * nNodes + node],
                                                             xLeft, xRight);
    }
}


cmlGPU::cmlGPU(int nNodes, int nSteps, float epsilon, float mu) {
    m_nNodes = nNodes;
    m_nSteps = nSteps;
    m_epsilon = epsilon;
    m_mu = mu;

    m_h_lattices = new float[nNodes * (nSteps + 1)];
    cudaMalloc((void **) &m_d_lattices, nNodes * (nSteps + 1) * sizeof(float));

}


cmlGPU::~cmlGPU()
{
    delete[] m_h_lattices;
    cudaFree(m_d_lattices);
}


void cmlGPU::init_lattice(int seed)
{
    std::srand(seed);
    for (int node = 0; node < m_nNodes; node++)
    {
        m_h_lattices[node] = ((float) rand()) / RAND_MAX;
    }
    cudaMemcpy(m_d_lattices, m_h_lattices, m_nNodes * (m_nSteps + 1) * sizeof(float), cudaMemcpyHostToDevice);
}


void cmlGPU::compute_lattice()
{
    // Set block and grid sizes

    int blocksInXGrid = max(1.0, ceil((float) m_nNodes / (float) BLOCK_SIZE));
    dim3 gridShape = dim3(blocksInXGrid);

    // Compute the CML
    computeCML<<<gridShape, BLOCK_SIZE>>>(m_d_lattices, m_nNodes, m_nSteps, m_epsilon, m_mu);
    cudaDeviceSynchronize();
}


void cmlGPU::export_evolution(std::string fileName)
{
    std::ofstream outputFile;
    outputFile.open(fileName);
    if (!outputFile.is_open())
    {
        std::cout << "The file did not opened properly. The CML cannot be exported..." << std::endl;
        return;
    }

    cudaMemcpy(m_h_lattices, m_d_lattices, m_nNodes * (m_nSteps + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    outputFile << "Number of nodes,Number of steps" << std::endl;
    outputFile << m_nNodes << "," << m_nSteps << std::endl;

    for (int step = 0; step <= m_nSteps; step++)
    {
        for (int node = 0; node < m_nNodes - 1; node++)
        {
            outputFile << m_h_lattices[step * m_nNodes + node] << ",";
        }
        outputFile << m_h_lattices[step * m_nNodes + m_nNodes - 1] << std::endl;
    }

    outputFile.close();
}