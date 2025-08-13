
#include <filesystem>
#include <string>
#include <random>

struct SimulationState {
    // Tunable parameters
    double packing_fraction;
    double density;
    double Lperiod;
    double Lperiod_perturb;
    double Amp;
    double Amp_in;
    double Amp_perturb;
    int nperiods;
    int nperiods_perturb;
    double mu;
    double phi;

    // System parameters
    double T;
    double radius;
    int d;
    double Lx, Ly;
    int N, N0;

    // MC parameters
    double step_size, step_size_xshift;
    int Neq, Nsim, Nupdate, Nprint;

    // Observables
    int Nbins;

    // Counters
    int Nacc_particle, Nmc_particle;
    int Nacc_insert, Nmc_insert;
    int Nacc_remove, Nmc_remove;
    int Nacc_xshift, Nmc_xshift;
    int count_rhoxavg;
    int count_rho_q;
    int Nmsd;
    double MSDx, MSDy, MSD;

    // Arrays (use std::vector or allocate dynamically)
    double* rhoxavg;
    double* c1;
    double* x;
    double* y;
    double* x00;
    double* y00;
    int* ix;
    int* iy;
    double* xx0;
    double* yy0;
    int* ix0;
    int* iy0;

    // Cell list
    int Ncell, Nmax;
    int* cell_list_index;
    int** cell_list;

    // Debugging
    // bool check0;
};


void run_simulation(SimulationState* state,std::ofstream& Nout,std::ofstream& Eta_out,const std::filesystem::path& dir,int file_count,std::mt19937& rng,const std::filesystem::path& pos_dir,const std::filesystem::path& output_dir); 
void pbc(SimulationState* state,double &x, double &y, int &ix, int &iy);
void pbc_distance(SimulationState* state,double &x, double &y);
int lattice_packing(SimulationState* state);
void mc_particle_move(SimulationState* state,std::mt19937 &rng);
void mc_xshift(SimulationState* state,std::mt19937 &rng);
void mc_step(SimulationState* state,std::mt19937 &rng);
void mc_swap(SimulationState* state,std::mt19937 &rng);
void mc_insert(SimulationState* state,std::mt19937 &rng);
void mc_remove(SimulationState* state,std::mt19937 &rng);
void GC_mc_swap(SimulationState* state,std::mt19937 &rng); 

void update_rhoxavg (SimulationState* state);
void update_rhoxavg_2D(SimulationState* state);

int determine_overlap(SimulationState* state,double xnew, double ynew, int i);
int determine_global_overlap_naive(SimulationState* state);
void create_cell_lists(SimulationState* state);
void update_cell_lists(SimulationState* state,int i, double x_old, double y_old, double x_new, double y_new);


void check_particle_0(SimulationState* state);
bool check_cell_lists(SimulationState* state);

void print_gr(SimulationState* state,int step);
void print_positions(SimulationState* state,const std::filesystem::path& dir,int step);
void print_rhox_avg_c1(SimulationState* state,const std::filesystem::path& dir,int file_count);

void print_N(SimulationState* state,const char *cstr,int step);
void print_eta(SimulationState* state,const char *cstr, double packing_fraction, double mu, double density);


double tilted_potential(SimulationState* state,double x,double y); 
std::string get_timestamp_string(); 

// debugging

