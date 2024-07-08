#ifndef CMLGPU_H
#define CMLGPU_H

#include <string>

class cmlGPU
{
public:
    // Constructor
    cmlGPU(int nNodes, int nSteps, float epsilon, float mu);
    // Destrcutor
    ~cmlGPU();

    // Initilialize the lattice
    void init_lattice(int seed);

    // Compute the evolution of the CML from the initial distribution
    void compute_lattice();

    // Export in a txt file the CML evolution
    void export_evolution(std::string fileName);


private: // Members
    float *m_d_lattices;
    float *m_h_lattices;
    int m_nNodes;
    int m_nSteps;
    float m_epsilon;
    float m_mu;
};

#endif // CMLGPU_H