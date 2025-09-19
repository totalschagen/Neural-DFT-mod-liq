#include <sstream>
#include <fstream>
#include <omp.h>
#include <filesystem>
#include <random>
#include <iostream>
#include <string>


#include "GCMC.h"
#include "nrutil.h"


namespace fs = std::filesystem;

int main() {

//  bool CL_check;
   


  std::string timestamp = get_timestamp_string();
  fs::path output_root = "/home/c705/c7051233/scratch_neural/data_generation/Output";
  fs::path density_profiles_root = "/home/c705/c7051233/scratch_neural/data_generation/Density_profiles";
  fs::path positions_root = "/home/c705/c7051233/scratch_neural/data_generation/Positions";

  fs::path output_dir = output_root / timestamp;
  fs::path density_profiles_dir = density_profiles_root / timestamp;
  fs::path positions_dir = positions_root / timestamp;

  fs::create_directories(output_dir);
  fs::create_directories(density_profiles_dir);
  fs::create_directories(positions_dir);
//////////////////////////////////////////////////////////////////////
 //prepare file for packing fraction and particle number  


  std::ofstream fout1(output_dir / "packing_fraction.dat");
  fout1 << "eta ; mu ; rho \n";


  std::ofstream fout2(output_dir / "N.dat");
  fout2 << "step N rho mu\n";

  ////////////////////////////////////////////////////////////////////
  // tunable parameters

  int nperiodsarray[]={10,20};
  double muarray[]={0.5,4.0,8.0};
  double amp_array[]={0.1,0.5};
  int total = 9;
  #ifndef BULK
  #pragma omp parallel for
  for(int m=0;m<12;m++){
    SimulationState* state = new SimulationState();

    state->Neq = 100000000;
    state->Nsim = 500000001;
    // state->Neq = 1000;
    // state->Nsim = 5001;

    // create thread local random number generator
    //std::random_device rd;
    std::mt19937 rng(m); // use the last digits of the seed and the loop index to create a unique seed for each thread
    // specify fixed simulation parameters

    // fix resolution
    double res = 50;
    state->T = 1.0;
    state->radius = 0.5;

    // fix system size
    int L_min = 8;
    int L_max = 18;

    std::uniform_int_distribution<int> Ldist(L_min,L_max);
    // state->Lx = double (Ldist(rng)); 
    state->Lx=10.0; // fixed system size
    state->Ly = state->Lx;

    // fix bounds for random parameters
    double mu_min = -2.0;
    float rr = 2*state->radius;
    int LLL_min = int(state->Lx / 2*rr);
    int LLL_max = int(state->Lx / 0.4*rr);  
    double mu_max = 5.0;

    // state->Nbins = LLL_max*20; // bins for 1 period
    state->Nbins = state->Lx * res; // bins with fixed resolution
    // initialize random distributions for potential variations
    // std::uniform_real_distribution<double> mudist(mu_min, mu_max);
    // std::uniform_real_distribution<double> dist(0.0, 1.0);
    // std::uniform_int_distribution<int> rand(LLL_min,LLL_max);
    
    state->mu = muarray[m%3];
    state->Amp_in = amp_array[(m/3)%2];
    state->Amp_perturb = 0.0;
    state->nperiods = nperiodsarray[(m/3)/2]; 
    state->nperiods_perturb = 0.0;
    //state->mu = 7.9;
    state->density = 3*0.1;
    printf("Amplitude = %f \n", state->Amp_in);
    printf("Perturbation-Amplitude = %f \n", state->Amp_perturb);
    printf("Number of periods = %d \n", state->nperiods);
    printf("Number of periods of perturbation_pot = %d \n", state->nperiods_perturb);

    state-> Lperiod = double(state->Lx*1.0) / double(state->nperiods*1.0);
    state-> Lperiod_perturb = 1;

    state->nperiods = int((state->Lx + 0.000000001) / state->Lperiod);
    state->nperiods_perturb = 0;

    run_simulation(state,fout2,fout1,density_profiles_dir,m,rng, positions_dir,output_dir); 

  }
  #endif
  #ifdef BULK
  #pragma omp parallel for   
  for(int m=1;m<50;m++){
    SimulationState* state = new SimulationState();

    // create thread local random number generator
    //std::random_device rd;
    std::mt19937 rng(m); // use the last digits of the seed and the loop index to create a unique seed for each thread
    state->T = 1.0;
    state->radius = 0.5;
    state->Lx = SLX;
    state->Ly = state->Lx;
    state->mu = -2.0 + 14.0 * m / 50.0;
    state->density = 2*0.1;
    run_simulation(state,fout2,fout1,density_profiles_dir,m,rng); 
    state->density = 4*0.1;
    run_simulation(state,fout2,fout1,density_profiles_dir,m,rng); 
  }
  #endif

  std::cout << timestamp << std::endl;
}