#ifndef CMLCPU_H
#define CMLCPU_H

#include <string>

// Computes the logistic map for a determined mu and x values
double logistic_map(float mu, float x);

// Computes the CML for a determined parameters mu and epsilon for a value and its neighbors.
float coupledmaplattice(float epsilon, float mu, float x, float xLeft, float xRight);

class cmlCPU
{
public:
    // Constructor
    cmlCPU(int nNodes, int nSteps, float epsilon, float mu);
    // Destrcutor
    ~cmlCPU();

    // Initilialize the lattice
    void init_lattice(int seed);

    // Compute the evolution of the CML from the initial distribution
    void compute_lattice();

    // Export in a txt file the CML evolution
    void export_evolution(std::string fileName);


private: // Members
    float *m_lattices;
    int m_nNodes;
    int m_nSteps;
    float m_epsilon;
    float m_mu;
};

#endif // CMLCPU_H