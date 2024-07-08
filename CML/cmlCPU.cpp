#include "cmlCPU.h"
#include <cstdlib>
#include <fstream>
#include <iostream>


double logistic_map(float mu, float x)
{
    return mu * x * (1- x);
}


float coupledmaplattice (float epsilon, float mu, float x, float xLeft, float xRight)
{
    return (1 - epsilon) * logistic_map(mu, x) + (epsilon / 2) * (logistic_map(mu, xRight) + logistic_map(mu, xLeft));
}


cmlCPU::cmlCPU(int nNodes, int nSteps, float epsilon, float mu) {
    m_nNodes = nNodes;
    m_nSteps = nSteps;
    m_epsilon = epsilon;
    m_mu = mu;

    m_lattices = new float[nNodes * (nSteps + 1)];
}


cmlCPU::~cmlCPU()
{
    delete[] m_lattices;
}


void cmlCPU::init_lattice(int seed)
{
    std::srand(seed);
    for (int node = 0; node < m_nNodes; node++)
    {
        m_lattices[node] = ((float) rand()) / RAND_MAX;
    }
}


void cmlCPU::compute_lattice()
{
    for (int step = 1; step <= m_nSteps; step++)
    {
        for (int node = 0; node < m_nNodes; node++)
        {
            float xLeft, xRight;
            // Apply Periodic Boundary Conditions

            xLeft = m_lattices[(step - 1) * m_nNodes + (node == 0 ? m_nNodes - 1 : node - 1)];
            xRight = m_lattices[(step - 1) * m_nNodes + (node == m_nNodes - 1 ? 0 : node + 1)];

            m_lattices[step * m_nNodes + node] = coupledmaplattice(m_epsilon, m_mu,
                                                                   m_lattices[(step - 1) * m_nNodes + node],
                                                                   xLeft, xRight);
        }
    }
}


void cmlCPU::export_evolution(std::string fileName)
{
    std::ofstream outputFile;
    outputFile.open(fileName);
    if (!outputFile.is_open())
    {
        std::cout << "The file did not opened properly. The CML cannot be exported..." << std::endl;
        return;
    }

    outputFile << "Number of nodes,Number of steps" << std::endl;
    outputFile << m_nNodes << "," << m_nSteps << std::endl;

    for (int step = 0; step <= m_nSteps; step++)
    {
        for (int node = 0; node < m_nNodes - 1; node++)
        {
            outputFile << m_lattices[step * m_nNodes + node] << ",";
        }
        outputFile << m_lattices[step * m_nNodes + m_nNodes - 1] << std::endl;
    }

    outputFile.close();
}
