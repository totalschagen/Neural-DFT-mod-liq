// compile with: g++ MC_hard_disks.cpp nrutil.cpp -w -O2 -DsSEED_IN=23451 -DAmp_in=1.7 -DSLX=30 -DRHO=0.7 -DLLL=10 -o ./MC_run.exe
#include "GCMC.h"

#include <math.h>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <csignal>
#include <omp.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include "nrutil.h"



namespace fs = std::filesystem;
#define PI 3.14159265358979323846264338

//#define BULK

long long rnumber;
int rnd;



void run_simulation(SimulationState* state,std::ofstream& Nout,std::ofstream& Eta_out,const fs::path& dir,int file_count,std::mt19937& rng,const fs::path& pos_dir) {
  clock_t begin = clock();
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);  
  timeinfo = localtime(&rawtime);
  printf("Strated at local time and date: %s", asctime(timeinfo));


  bool CL_check;

  
  state->Amp = 0.0;//initially for mixing

  ////////////////////////////////////////////////////////////////////

  // system parameters which follow from the tunable parameters
  state->N = int(state->Lx * state->Ly * state->density + 0.000000000001);
  state->N0 = 1000;
  state->packing_fraction = state->density * PI * state->radius * state->radius;

  // MC simulation parameters
  state->step_size = 0.01;
  state->step_size_xshift = 0.01;
  int N_mixing = 20000;   

  state->Nprint = 5000;
  // int N_mixing = 200;   
  // Nprint = 100;
  // Neq = 100;
  // Nsim = 1000;
  state->Nupdate = 100;
  int t0 = 0;
  state->Nmsd = state->Nsim / state->Nprint;

  // Observable rho(x)
  printf("Number of bins: %d \n", state->Nbins);
  ////////////////////////////////////////////////////////////////////
  ////////////////// NO CHANGES IN THE FOLLOWING ! ///////////////////
  ////////////////////////////////////////xrandr --output None-1 --right-of eDP-1

  // arrays initialization
  state->rhoxavg = dvector(0, (state->Nbins )*(state->Nbins)-1);
  for (int i = 0; i <state->Nbins*(state->Nbins-1); i++) {
    state->rhoxavg[i] = 0.0;
  }

  state->x = dvector(0, state->N0 - 1);
  state->y = dvector(0, state->N0 - 1);
  state->ix = ivector(0, state->N0 - 1);
  state->iy = ivector(0, state->N0 - 1);
  //xx0 = dvector(0, N - 1);
  //yy0 = dvector(0, N - 1);
  //ix0 = ivector(0, N - 1);
  //iy0 = ivector(0, N - 1);


  // system initialization
  int check_initial_configuration = lattice_packing(state); // initialize particles
    //~ print_positions(0);

  //print_positions(state,0);

  // prepare output files

  ////////////////////////////////////////////////////////////////////
  /////////////// INITIALIZATION FINISHED ! //////////////////////////
  ////////////////////////////////////////////////////////////////////
  // perform simulation
  // mixing
	printf("|---------------------|\n");
	printf("|   mixing  started!  |\n");
	printf("|---------------------|\n");
	for(int im=1; im<=N_mixing; im++) {
		mc_swap(state,rng);
		if(im%1000==0) {
      printf("mixing step\n: %d", im);
      //check_particle_0();
    // #pragma omp critical
    //   {
    //     Nout << im << " " << state->N << " " << state->density << " " << state->mu << "\n"; 
    //   }
    }
  }
	printf("\n");
	//print_positions(1);
	printf("|---------------------|\n");
	printf("|   mixing finished!  |\n");
	printf("|---------------------|\n");
  state->Amp = (state->Amp_in * 1.0);	
  #ifdef BULK
  state->Amp = 0.0;
  #endif
  #ifdef TESTGC
  // test grand canonical steps
  int initial_N = N;
  printf("Initial particle number: %d \n", N);
  printf("chemical potential: %f \n", mu);
  //check_cell_lists();
  for (int i = 0; i < 1000; i++) {
    if(i % 100 == 0) printf("test-step: %d \n", i);
    //check_cell_lists();
    mc_insert();
    mc_remove(); 
  }

  printf("Initial particle number: %d \n", initial_N);
  printf("Final particle number: %d \n", N);
  printf("Removal acceptance ratio: %f \n",Nacc_remove / ((double)Nmc_remove));
  printf("Insertion acceptance ratio: %f \n",Nacc_insert / ((double)Nmc_insert));
  exit(0);
  #endif
	// perform simulation
	// Equilibration
  for(int i=1; i<state->Neq; ++i)
  {
    if(i%1000==0) {
      printf("equ step: \n%d", i);
      // #pragma omp critical
      // {
      //   Nout << i << " " << state->N << " " << state->density << " " << state->mu << "\n"; 
      // }
      //check_particle_0();
    }
    if (i % state->Nupdate == 0) {
      if (i != 0) {
        if (state->Nacc_particle / ((double)state->Nmc_particle) < 0.45) {
          state->step_size *= 0.9;
          state->Nacc_particle = state->Nmc_particle = 0;
        } else if (state->Nacc_particle / ((double)state->Nmc_particle) > 0.55) {
          if (state->step_size * 1.1 <= state->Ly * 0.5 - 0.1)
            state->step_size *= 1.1;
          state->Nacc_particle = state->Nmc_particle = 0;
        }
        if (state->Nacc_xshift / ((double)state->Nmc_xshift) < 0.45) {
          state->step_size_xshift *= 0.9;
          state->Nacc_xshift = state->Nmc_xshift = 0;
        }
        if (state->Nacc_xshift / ((double)state->Nmc_xshift) > 0.55) {
          if (state->step_size_xshift * 1.1 < 0.5)
            state->step_size_xshift *= 1.1;
          state->Nacc_xshift = state->Nmc_xshift = 0;
        }
        state->Nacc_particle = state->Nmc_particle = 0;
        state->Nacc_xshift = state->Nmc_xshift = 0;
        if (i%100==0){
          //print_N(cstr,i);
        }
      }
    }
    GC_mc_swap(state,rng);

  }
  printf("\n");
  printf("|---------------------|\n");
	printf("|   equili finished!  |\n");
	printf("|---------------------|\n");
  for (int i = 1; i <= state->Nsim; i++) {
    if (i % state->Nprint == 0) {
      printf(" %d steps done. \n", i);
    }
    if (i%100==0){
      #ifdef BULK
      #pragma omp critical
      {
        Nout << i << " " << state->N << " " << state->density << " " << state->mu << "\n"; 
      }
      #endif
      update_rhoxavg_2D(state);
      //print_N(N_print,i);
    }
    if (i%1000==0){
      printf("sim step: \n%d", i);

    }
    GC_mc_swap(state,rng);

    if (i % state->Nprint == 0) {
      //print_positions(i);
      #pragma omp critical
      {
        Nout << i << " " << state->N << " " << state->density << " " << state->mu << "\n"; 
      }
      CL_check=check_cell_lists(state);
      if (CL_check == false) {
        printf("Cell list check failed! \n");
        for (int j = 0; j < state->Nmax; j++) {
          printf("Cell list: %d %d  \n", state->cell_list_index[0], state->cell_list[state->cell_list_index[0]][j]);
        }
      }
      #ifdef BULK
      #pragma omp critical
      {
        Eta_out << state->packing_fraction << " " << state->mu << " " << state->density <<"\n"; 
      }
      
      #endif
      //print_eta(state,eta_print,state-> packing_fraction, state->mu, state->density);
      if (determine_global_overlap_naive(state) == 1) {
        printf("error overlap %d \n", i);
        break;
      }
    }
  }
  #ifndef BULK
  printf("Writing density profile to file:  \n");

  state->packing_fraction = state->N /(state->Lx*state->Ly) * PI * state->radius * state->radius;
  print_rhox_avg_c1(state,dir,file_count);
  #endif
  printf("\n");
  print_positions(state,pos_dir,file_count);
  printf("Packing fraction = %f \n", state->packing_fraction);
  printf("Period = %f \n", state->Lperiod);
  printf("Amplitude = %f \n", state->Amp);
  printf("Number of particles = %d \n", state->N);
  printf("Simulation Box = %f %f \n", state->Lx, state->Ly);
  printf("Number of periods = %d \n", state->nperiods);
  //printf("Wavenumber grid size = %d x %d \n", Mq, Mq);
  //printf("Wavenumber Delta = %f \n", Deltaq);
  printf("Check! Effective number density %f \n", state->N / state->Lx / state->Ly);
  clock_t end = clock();
  double time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;
  printf("%ld %ld, spent: %f\n", begin, end, time_spent);
  //
  time(&rawtime);
  printf(" Strated at local time and date: %s", asctime(timeinfo));
  timeinfo = localtime(&rawtime);
  printf("Finished at local time and date: %s", asctime(timeinfo));
 }



/********************************************************************/
int lattice_packing(SimulationState* state) {

  int nx = int( state->Lx / (2 * state->radius)); // number of particle that fit in the x direction on a line
  int ny = int(state->N / nx) + 1; // number of lines
  if (state->N % nx == 0)
    ny = ny - 1;
  double epsy = state->Ly / ny; // average distance between lines
  double epsy2 = 0.333 * (epsy - sqrt(3.) * state->radius);

  for (int index = 0; index < state->N; index++) {
    int j = index / nx;
    int i = index % nx;
    double xloc = 0.0;
    double yloc = 0.0;
    if (j % 2 == 0) {
      xloc = 2 * state->radius * (0.499 + i);
      yloc = 2 * state->radius * (j + 0.499) * epsy;
      if (i % 2 == 1)
        yloc += epsy2;
    } else {
      xloc = 2 * state->radius * (0.999 + i);
      yloc = 2 * state->radius * (j + 0.5) * epsy;
      if (i % 2 == 1)
        yloc += epsy2;
    }
    pbc_distance(state,xloc, yloc);
    state->x[index] = xloc;
    state->y[index] = yloc;
    state->ix[index] = 0;
    state->iy[index] = 0;
  }
  printf("Packing fraction = %f \n", state->packing_fraction);
  printf("Period = %f \n", state->Lperiod);
  printf("Amplitude = %f \n", state->Amp);
  printf("Number of particles = %d \n", state->N);
  printf("Simulation Box = %f %f \n", state->Lx, state->Ly);
  printf("Number of periods = %d \n", state->nperiods);
  printf("Chemical potential = %f \n", state->mu);
  //printf("Wavenumber grid size = %d x %d \n", Mq, Mq);
  //printf("Wavenumber Delta = %f \n", Deltaq);
  printf("Check! Effective number density %f \n", state->N / state->Lx / state->Ly);

  // cell list
  state->Ncell = int(state->Lx / (3 * state->radius)); // Should be a meaningful number
  state->Nmax = 200;
  state->cell_list_index = ivector(0, state->N0 - 1);
  state->cell_list = imatrix(0, (state->Ncell * state->Ncell) - 1, 0, state->Nmax - 1);
  create_cell_lists(state);
  printf("Number of particles prescibed: %d, Number of particles: %d \n", state->N0, state->N);
  //check_cell_lists();
  //check_particle_0();

  printf("Check! Effective packing fraction %f \n",
         state->N * PI * state->radius * state->radius / state->Lx / state->Ly);
  printf("---------------------------------\n");

  if (determine_global_overlap_naive(state) == 1) {
    printf("initial overlap %d \n", 1);
    return 1;
  } else {
    return 0;
  }
}

// consider periodic boundary conditions
void pbc(SimulationState* state,double &x, double &y, int &ix, int &iy) {
if (x >= state->Lx / 2.0) {
    x -= state->Lx;
    ix += 1;
  }
  if (x < -state->Lx / 2.0) {
    x += state->Lx;
    ix -= 1;
  }
  if (y >= state->Ly / 2.0) {
    y -= state->Ly;
    iy += 1;
  }
  if (y < -state->Ly / 2.0) {
    y += state->Ly;
    iy -= 1;
  }
}

void pbc_distance(SimulationState* state,double &x, double &y) {
  if (x >= state->Lx / 2.0)
    x -= state->Lx;
  if (x < -state->Lx / 2.0)
    x += state->Lx;
  if (y >= state->Ly / 2.0)
    y -= state->Ly;
  if (y < -state->Ly / 2.0)
    y += state->Ly;
}

// initialize cell lists
void create_cell_lists(SimulationState* state) {
  for (int a = 0; a < state->Ncell; a++) {
    for (int b = 0; b < state->Ncell; b++) {
      for (int j = 0; j < state->Nmax; j++) {
        state->cell_list[a * state->Ncell + b][j] = -1;
      }
    }
  }

  // find cell lists for particles
  for (int i = 0; i < state->N; i++) {
    int xint = (int)(state->Ncell * (state->x[i] / state->Lx + 0.5));
    int yint = (int)(state->Ncell * (state->y[i] / state->Ly + 0.5));

    if (xint < 0)
      xint = 0;
    if (yint < 0)
      yint = 0;
    if (xint > (state->Ncell - 1))
      xint = state->Ncell - 1;
    if (yint > (state->Ncell - 1))
      yint = state->Ncell - 1;

    state->cell_list_index[i] = xint * state->Ncell + yint;

    bool particle_stored = false;
    for (int j = 0; j < state->Nmax; j++) {
      if (state->cell_list[xint * state->Ncell + yint][j] == -1) {
        state->cell_list[xint * state->Ncell + yint][j] = i;
        particle_stored = true;
        break;
      }
    }
    if (!particle_stored) {
      printf("ERROR: cell list storage exceeded!!! Nmax must be increased! \n");
      std::raise(SIGINT);
    }
  }
}

void update_cell_lists(SimulationState* state, int i, double x_old, double y_old, double x_new,
                       double y_new) {
  // update cell lists for particle i
  int cell_number_old = state->cell_list_index[i];  

  int xint_new = (int)(state->Ncell * (x_new / state->Lx + 0.5));
  int yint_new = (int)(state->Ncell * (y_new / state->Ly + 0.5));
  int cell_number_new = xint_new * state->Ncell + yint_new;

  if (cell_number_old != cell_number_new) {
    state->cell_list_index[i] = cell_number_new;
    int k = -1;
    for (int j = 0; j < state->Nmax; j++) {
      if (state->cell_list[cell_number_old][j] == i) {
        k = j;
        break;
      }
    }
    if (k == -1 || k >= state->Nmax) {
      create_cell_lists(state);
    } else {
      for (int j = k; j < state->Nmax - 1; j++) {
        state->cell_list[cell_number_old][j] = state->cell_list[cell_number_old][j + 1];
      }
      state->cell_list[cell_number_old][state->Nmax - 1] = -1;
      for (int j = 0; j < state->Nmax; j++) {
        if (state->cell_list[cell_number_new][j] == -1) {
          state->cell_list[cell_number_new][j] = i;
          break;
        }
      }
    }
  }
}

int determine_overlap(SimulationState* state,  double xnew, double ynew, int i) {
  double sigma2 = 4 * state->radius * state->radius;
  // determine x,y,z cell index
  // int cell_list_indexi = state->cell_list_index[i]; // this is the old position
  int xint = (int)(state->Ncell * (xnew / state->Lx + 0.5));
  int yint = (int)(state->Ncell * (ynew / state->Ly + 0.5));
  if (xint < 0)
    xint = 0;
  if (yint < 0)
    yint = 0;
  if (xint > (state->Ncell - 1))
    xint = state->Ncell - 1;
  if (yint > (state->Ncell - 1))
    yint = state->Ncell - 1;
  int cell_list_indexi = xint * state->Ncell + yint;

  int cell_list_y = cell_list_indexi % state->Ncell;
  int cell_list_x = ((cell_list_indexi - cell_list_y) / state->Ncell);
  ////////////////////////////////////
  for (int a = cell_list_x - 1; a <= cell_list_x + 1; a++) {
    for (int b = cell_list_y - 1; b <= cell_list_y + 1; b++) {
      int aloc = a;
      int bloc = b;
      if (aloc < 0)
        aloc += state->Ncell;
      if (aloc >= state->Ncell)
        aloc -= state->Ncell;
      if (bloc < 0)
        bloc += state->Ncell;
      if (bloc >= state->Ncell)
        bloc -= state->Ncell;

      for (int j = 0; j < state->Nmax; j++) {
        int jloc = state->cell_list[aloc * state->Ncell + bloc][j];

        if (jloc != -1) {
          if (jloc != i) {
            // printf("%d\n",j);
            double dxloc = xnew - state->x[jloc];
            double dyloc = ynew - state->y[jloc];
            pbc_distance(state,dxloc, dyloc);
            double r2 = dxloc * dxloc + dyloc * dyloc;
            if (r2 < sigma2) {
              return 1;
            }
          }
        } else {
          break;
        }
      }
    }
  }
  return 0;
}

int determine_global_overlap_naive(SimulationState* state) {
  double sigma2 = 4 * state->radius * state->radius;
  for (int i = 0; i < state->N; i++) {
    for (int j = 0; j < state->N; j++) {
      if (i != j) {
        double dxloc = state->x[i] - state->x[j];
        double dyloc = state->y[i] - state->y[j];
        pbc_distance(state,dxloc, dyloc);
        double r2 = dxloc * dxloc + dyloc * dyloc;
        if (r2 < sigma2 - 0.00000000001) {
          printf("pos %d %f %f\n", i, state->x[i], state->y[i]);
          printf("pos: %d %f %f\n", j,state-> x[j], state->y[j]);
          printf("distance: %f\n", r2);
          return 1;
        }
      }
    }
  }
  return 0;
}

/********************************************************************/
void mc_particle_move(SimulationState* state, std::mt19937& rng) {
  // move a ramdom particle in a random direction
  std::uniform_int_distribution<int> int_rand(0, state->N - 1);
  int i = int_rand(rng);
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  double dphi = float_dist(rng) * 2.0 * PI;
  double dx = state->step_size * cos(dphi);
  double dy = state->step_size * sin(dphi);
  double xnew = state->x[i] + dx;
  double ynew = state->y[i] + dy;
  int ixnew = state->ix[i];
  int iynew = state->iy[i];
  pbc(state,xnew, ynew, ixnew, iynew);
  // determine overlap
  int overlap = 0;
  overlap = determine_overlap(state,xnew, ynew, i);
  // if no overlap: move particle and update cell list
  if (overlap == 0) {
    // check potential in x direction
    double Vold = state->Amp * cos(2 * PI * state->x[state->N] / state->Lperiod);
    Vold = tilted_potential(state,state->x[i], state->y[i]);
    double Vnew = state->Amp * cos(2 * PI * xnew / state->Lperiod);
    Vnew = tilted_potential(state,xnew, ynew);
    double arg = -(Vnew - Vold) / state->T;
    double rr2 = float_dist(rng);
    if (exp(arg) > rr2) {
      double xold = state->x[i];
      double yold = state->y[i];
      state->x[i] = xnew;
      state->y[i] = ynew;
      state->ix[i] = ixnew;
      state->iy[i] = iynew;
      int CL_index = state->cell_list_index[i];
      update_cell_lists(state,i, xold, yold, xnew, ynew);
      state->Nacc_particle++;
      //~ printf("index: %d %d  \n",Nacc_particle, Nmc_particle);
      //~ printf("index: %d %f %f  \n",i,xold,yold);
      //~ printf("index: %d %f %f  \n",i,xnew,ynew);
      //~ printf("%f %f %f %f \n",Vnew, Vold, exp(arg), rr2);
      /*
      if(check0==true){
        if(i==0){
          printf("Particle 0 got moved from CL %d to CL %d \n", CL_index, state->cell_list_index[i]);
          printf("Particle 0 got moved from %f %f to %f %f \n", xold, yold, xnew, ynew);
        }
      }
      */
    }
  }
  state->Nmc_particle++;
}

void mc_xshift(SimulationState* state, std::mt19937& rng) {
  // move all particles in x direction
  int dir = -1;
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  double rans = float_dist(rng);
  if (rans > 0.5)
    dir = 1;
  double dx = dir * state->step_size_xshift;
  double Vold = 0.;
  double Vnew = 0.;
  for (int i = 0; i < state->N; i++) {
    double xnew = state->x[i] + dx;
    // Vold += state->Amp * cos(2 * PI * state->x[i] / state->Lperiod);
    // Vnew += state->Amp * cos(2 * PI * xnew / state->Lperiod);
    Vold += tilted_potential(state,state->x[i], state->y[i]);
    Vnew += tilted_potential(state,xnew, state->y[i]);
  }
  double arg = -(Vnew - Vold) / state->T;
  double rr2 = float_dist(rng);
  if (exp(arg) > rr2) {
    for (int i = 0; i < state->N; i++) {
      double xnew = state->x[i] + dx;
      double ynew = state->y[i];
      int ixnew = state->ix[i];
      int iynew = state->iy[i];
      pbc(state,xnew, ynew, ixnew, iynew);
      state->x[i] = xnew;
      state->y[i] = ynew;
      state->ix[i] = ixnew;
      state->iy[i] = iynew;
    }
    //if(check0==true) printf("particle 0 was in cell %d \n", state->cell_list_index[0]);
    create_cell_lists(state);
    //if(check0==true) printf("particle 0 got shifted to cell %d \n", state->cell_list_index[0]);
    state->Nacc_xshift++;
  }
  state->Nmc_xshift++;
}

void mc_insert(SimulationState* state, std::mt19937& rng) {
  // insert a new particle
  /*
  if(check0== true){
    if(N==0){
      printf("No Particles in the system \n");
    }
  }
  */
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  double xnew = float_dist(rng) * state->Lx;
  double ynew = float_dist(rng) * state->Ly;
  pbc_distance(state,xnew, ynew);
  int ixnew = 0;
  int iynew = 0;
  pbc(state,xnew, ynew, ixnew, iynew);
 // determine overlap
  int overlap = 0;
  overlap = determine_overlap(state,xnew, ynew, state->N+1);
  // if no overlap: add particle and update cell list
  if (overlap == 0) {
    // check potential in x direction
//    double Vnew = state->Amp * cos(2 * PI * xnew / state->Lperiod);
    double Vnew = tilted_potential(state,xnew, ynew);
    double arg = (state->mu-Vnew ) / state->T;
    double rr2 = float_dist(rng);
    if ((state->Lx*state->Ly)/(state->N+1)*exp(arg) > rr2) {    // if move is accepted, update N to accord for longer particle array
      state->x[state->N] = xnew;
      state->y[state->N] = ynew;
      state->ix[state->N] = ixnew;
      state->iy[state->N] = iynew;
      int xint = (int)(state->Ncell * (state->x[state->N] / state->Lx + 0.5));
      int yint = (int)(state->Ncell * (state->y[state->N] / state->Ly + 0.5));

      
      int c_index = xint * state->Ncell + yint;
      state->cell_list_index[state->N] = c_index;
      bool particle_stored = false;
      for (int j = 0; j < state->Nmax; j++) {
        /*
        if(check0==true){
          if (state->cell_list[c_index][j] == 0) {
            printf("ERROR: particle %d in manipulated cell list %d \n", 0, c_index);
          }
        }
        */
        if (state->cell_list[c_index][j] == -1) {
          state->cell_list[c_index][j] = state->N;
          particle_stored = true;
          break;
        }
      }
      if (!particle_stored) {
        printf("ERROR: cell list storage exceeded!!! Nmax must be increased! \n");
      }
/* 
      bool check = check_cell_lists();
      if (check == false) {
        printf("inserted particle %d %d  \n", N, state->cell_list_index[N] );
        printf("Ncell : %d xint:%d yint: %d  \n",Ncell, xint, yint );
        for(int j = 0; j < Nmax; j++) {
          printf("Inserted particle in Cell list: %d which holds %d  \n", c_index, state->cell_list[c_index][j]);
          if(state->cell_list[c_index][j] == -1) break;
        }
      }

 */      //check_particle_0();
      state->N++;
      state->Nacc_insert++;

      //~ printf("index: %d %d  \n",Nacc_particle, Nmc_particle);
      //~ printf("index: %d %f %f  \n",i,xold,yold);
      //~ printf("index: %d %f %f  \n",i,xnew,ynew);
      //~ printf("%f %f %f %f \n",Vnew, Vold, exp(arg), rr2);
    }
  } 
  state->Nmc_insert++;

}

void mc_remove(SimulationState* state, std::mt19937& rng) {
  // remove a particle
  // select a random particle
  std::uniform_int_distribution<int> int_rand(0, state->N - 1);
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  int i = int_rand(rng);
  // check energy of removal
  // double Vnew = state->Amp * cos(2 * PI * state->x[i] / state->Lperiod);
  double Vnew = tilted_potential(state,state->x[i], state->y[i]);
  double arg = (Vnew-state->mu) /state-> T;
  double rr2 = float_dist(rng);
  if ( state->N/(state->Lx * state->Ly ) * exp(arg) > rr2) { // if move is accepted, update N to accord for shorter particle array
    // remove particle and update cell list
    int index;
    int old_index;
    index = state->cell_list_index[i];
    old_index = index;
    // update cell list of removed particle
    bool particle_stored = true;
    int particle_index_in_cell = -1;
    for (int j = 0; j < state->Nmax; j++) {
      //find position of removed particle in it's cell list
      if (state->cell_list[index][j] == i) {
        particle_index_in_cell = j;
      }
      if (state->cell_list[index][j+1] == -1) {
        // find last position in cell list
        // replace removed particle with last particle in cell list
        state->cell_list[index][particle_index_in_cell]=state->cell_list[index][j];
        state->cell_list[index][j] = -1;
        particle_stored = false;
        break;
      } 
    }
    if (particle_stored) {
      printf("ERROR: PARTICLE REMOVAL UNSUCCESSFUL!!!  \n");
      //printf("index: %d %d  \n", i, N);
      //for (int j = 0; j < Nmax; j++) {
       // printf("Cell list: %d %d  \n", index, state->cell_list[index][j]);
      //}
      //exit(1);
    }
    // replace particle with last particle in particle array
    state->x[i] = state->x[state->N - 1];
    state->y[i] = state->y[state->N - 1];
    state->ix[i] = state->ix[state->N - 1];
    state->iy[i] = state->iy[state->N - 1];
    // update list of cell list of particles
    state->cell_list_index[i] = state->cell_list_index[state->N - 1];
    index = state->cell_list_index[i];
    // update the last entry
    state->x[state->N - 1] = 0.0;
    state->y[state->N - 1] = 0.0;
    state->ix[state->N - 1] = 0;
    state->iy[state->N - 1] = 0;
    // update cell list of particle (former last one in particle list) replacing removed particle
    for (int j = 0; j < state->Nmax; j++) {
      if (state->cell_list[index][j] == state->N-1) {
        state->cell_list[index][j] = i;
        break;
      } 
    }

    state->cell_list_index[state->N - 1] = -1;

    state->N--;
    state->Nacc_remove++;
    /*
    if(check0==true){
        if(i==0){
          printf("Particle 0 got removed from CL %d and was replaced by particle %d \n", old_index, N+1);
        }
      }
        */
    //bool check = check_cell_lists();
    //if(!check){
    //  printf("removed particle %d cell list %d  \n", i, index );
    //  printf("replaced with particle formerly at %d cell list %d  \n", N, state->cell_list_index[i] );
    //}
    //check_particle_0();
  }
  state->Nmc_remove++;
  
}
void mc_step(SimulationState* state, std::mt19937& rng) {
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  double rans = float_dist(rng);
  if (rans <= 1. / state->N) {
    mc_xshift(state,rng);
  } else {
    mc_particle_move(state,rng);
  }
}

void mc_swap(SimulationState* state, std::mt19937& rng) {
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  for (int k = 0; k < state->N; k++) {
    double rans = float_dist(rng);
    if (rans <= 1. / state->N) {
      mc_xshift(state,rng);
    } else {
      mc_particle_move(state,rng);
    }
  }
}

void GC_mc_swap(SimulationState* state,std::mt19937& rng) {
  std::uniform_real_distribution<double> float_dist(0.0,1.0);
  for (int k = 0; k < state->N; k++) {
    double rans = float_dist(rng);
    #ifndef BULK
    if (rans < 0.5){
      rans = float_dist(rng);
      if (rans <= 1. / state->N) {
        mc_xshift(state,rng);
      } else {
        mc_particle_move(state,rng);
      }
    }
    #endif
    #ifdef BULK
    if (rans < 0.5){
        mc_particle_move(state,rng);
      }
    #endif
     else{
      rans = float_dist(rng);
      if (rans <= 0.5) {
        mc_insert(state,rng);
        
      } else {
        mc_remove(state,rng);
        
      }
    }
  }

} /********************************************************************/
// Diagnostic functions

bool check_cell_lists(SimulationState* state){
  for(int i=0;i<state->N;i++){
    int index;
    bool check = false;
    index = state->cell_list_index[i];
    for(int j=0;j<state->Nmax;j++){
      if(state->cell_list[index][j]==i){
        check = true;
        return true;
      }
    }
    if(check==false){
        printf("ERROR: PARTICLE NOT FOUND IN CELL LIST!!!  \n");
        printf("Particle %d in Cell list: %d  \n", i, index);
        for(int k=0;k<state->Nmax;k++){
          printf("Cell list: %d has particle: %d  \n", index,state->cell_list[index][k] );
          if(state->cell_list[index][k]==i){
            printf("Found particle %d in cell list %d \n", i, index);
            break;
          }
          if(state->cell_list[index][k]==-1){
            break;
          }

        }
        for (int k =0;k<state->Ncell*state->Ncell-1;k++){
          for(int j=0;j<state->Nmax;j++){
            //printf("Cell list: %d has particle: %d  \n", k,state->cell_list[k][j] );
            if(state->cell_list[k][j]==i){
              printf("Found particle %d in cell list %d \n", i, k);
              break;
            }
            if(state->cell_list[k][j]==-1){
              break;
            } 
          }
        }
        exit(1);
        return false; 
    }
  }
  return true;
}

//Track particle 0
void check_particle_0(SimulationState* state){
  int p_index = 0;
  if(state->cell_list_index[p_index] == -1){
    printf("Particle 0 removed by MC-move \n");
    printf("Current state->N is %d\n",state->N);
    return;
  }
  //printf("Particle 0 thinks it is in cell list %d \n", state->cell_list_index[p_index]);
  for(int i =0;i<state->Nmax;i++){
    if(state->cell_list[state->cell_list_index[p_index]][i]==p_index){
   //   printf("Particle 0 found in cell list %d \n", state->cell_list_index[p_index]);
      return;
    }
    if(state->cell_list[state->cell_list_index[p_index]][i]==-1){
      break;
    }
  }
  printf("ERROR: PARTICLE 0 NOT FOUND IN CELL LIST!!!  \n");
  for(int i =0;i<state->Ncell;i++){
    for(int j=0;j<state->Nmax;j++){
      printf("Cell list: %d has particle: %d  \n", i,state->cell_list[i][j] );
      if(state->cell_list[i][j]==p_index){
        printf("Found particle %d in cell list %d \n", p_index, i);
        break;
      }
      if(state->cell_list[i][j]==-1){
        break;
      } 
    }
  }
  exit(1);
}
/********************************************************************/
// 1D density profile, relative to periodic potential
void update_rhoxavg(SimulationState* state) {
  double binsize = state->Lperiod / state->Nbins;
  for (int i = 0; i < state->N; i++) {
    double xt = state->x[i];
    double yt = state->y[i];
    int j = (int)(xt / state->Lperiod);
    xt = xt - j * state->Lperiod;
    if (xt >= state->Lperiod / 2.0 - binsize / 2.0)
      xt -= state->Lperiod;
    if (xt < -state->Lperiod / 2.0 - binsize / 2.0)
      xt += state->Lperiod;
    xt += state->Lperiod / 2.0 + binsize / 2.0;

    j = (int)(yt / state->Lperiod);
    yt = yt - j * state->Lperiod;
    if (yt >= state->Lperiod / 2.0 - binsize / 2.0)
      yt -= state->Lperiod;
    if (yt < -state->Lperiod / 2.0 - binsize / 2.0)
      yt += state->Lperiod;
    yt += state->Lperiod / 2.0 + binsize / 2.0;
    
    int xindex = (int)(xt / binsize);
    int yindex = (int)(yt / binsize);
    state->rhoxavg[xindex*state->Nbins+yindex] += 1. / (binsize * binsize* state->nperiods);
  }
   state->count_rhoxavg += 1;
}

// new 2D density grid for training data
void update_rhoxavg_2D(SimulationState* state){
  double binsize = state->Lx /state->Nbins;
  for (int i = 0; i < state->N; i++) {
    double xt = state->x[i];
    double yt = state->y[i];
    xt = fmod(fmod(xt, state->Lx) + state->Lx, state->Lx);
    yt = fmod(fmod(yt, state->Ly) + state->Ly, state->Ly);
    int ix = (int)(xt / binsize);
    int iy = (int)(yt / binsize);
    state->rhoxavg[ix*state->Nbins+iy] += 1.;
  }
  state->count_rhoxavg += 1;

}

void update_N_avg(){

}


/******************************* */
// Printing functions

/********************************************************************/
// calculate and print g(r)
void print_gr(SimulationState* state,int step) {

  // constants for histogram
  int Nhisto = 100;
  double drhisto = 0.05;

  // declare and initialize histogram
  double *histo = dvector(0, Nhisto - 1);
  for (int i = 0; i < Nhisto; i++) {
    histo[i] = 0.0;
  }

  for (int i = 0; i < state->N; i++) {
    double xi = state->x[i];
    double yi = state->y[i];

    for (int j = 0; j < state->N; j++) {
      if (j != i) {
        double dxloc = xi - state->x[j];
        double dyloc = yi - state->y[j];
        pbc_distance(state,dxloc, dyloc);

        double dr = sqrt(dxloc * dxloc + dyloc * dyloc);
        int drint = (int)(dr / drhisto + 0.5);

        // printf("dr %f drint %d\n",dr,drint);
        //  bin the particle distance into a histogram
        if (drint < Nhisto)
          histo[drint]++;
      }
    }
  }

  // write histogram data
  std::stringstream sstr;
  sstr << "gr_step" << step << ".dat";
  const std::string tmp = sstr.str();
  const char *cstr = tmp.c_str();

  FILE *out = fopen(cstr, "w");

  for (int i = 0; i < Nhisto; i++) {
    double A = PI * (pow(i * drhisto + drhisto, 2) - pow(i * drhisto, 2));
    fprintf(out, "%f %f\n", i * drhisto,
            histo[i] * state->Lx * state->Ly / ((double)state->N * state->N * A));
  }

  fclose(out);
}

// print particle positions
void print_positions(SimulationState* state,const fs::path& dir,int step) {
  std::string posi_file_name = "positions_step" + std::to_string(step) + ".dat";
  std::ofstream posi_file(dir / posi_file_name);

  for (int i = 0; i < state->N; i++) {
    // fprintf(out,"%f %f  %d %d\n",x[i],y[i],state->ix[i],iy[i]);
    posi_file <<i << " " <<state->x[i] << " " <<state->y[i]<< "\n"; //,ix[i],iy[i]);
  }

  posi_file.close();
}

// print density profile
void print_rhox_avg_c1(SimulationState* state,const fs::path& dir,int file_count) {
  std::string density_file_name = "rho_MC_2D_" + std::to_string(file_count) + ".dat";
  std::ofstream density_file(dir / density_file_name);
  density_file << "x y rho muloc " << state->nperiods << " " << state->mu <<" " << state->packing_fraction << " "<<state->Amp << " " << state->nperiods_perturb << " " << state->Amp_perturb <<" " << state->Lx << "\n";

  double binarea = (state->Lx / state->Nbins)*(state->Ly / state->Nbins);
  for (int i = 0; i < state->Nbins; i++) {
    for (int j = 0; j < state->Nbins; j++) {
      double xbin =state->Lx * (i + 0.5) / state->Nbins;
      double ybin =state->Ly * (j + 0.5) / state->Nbins;
      // double muloc =  state->Amp * cos(2 * PI * xbin / state->Lperiod)-state->mu;
      double muloc =  tilted_potential(state,xbin,ybin)-state->mu;
      density_file << xbin << " " << ybin << " " << state->rhoxavg[i*state->Nbins+j] / (state->count_rhoxavg*binarea) << " " << muloc << "\n";
    }
  }
  density_file.close();
}


void print_N(SimulationState* state,const char *cstr,int step){


  FILE *out = fopen(cstr, "a");
  fprintf(out,"%d  %d  %f  %f\n", step, state->N,state->density,state->mu); 
  fclose(out);
}

void print_eta(SimulationState* state,const char *cstr, double packing_fraction, double mu, double density){
  FILE *fp2 = fopen(cstr, "a");
  fprintf(fp2,"%f ; %f ; %f \n",packing_fraction, state->mu, state->density);
  fclose(fp2);
}


std::string get_timestamp_string() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << "parallel" <<std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
  return ss.str();
}



double tilted_potential(SimulationState* state,double x,double y) {
  // calculate potential at position x
  double V = state->Amp * cos(2 * PI * x / state->Lperiod);
  //double V_perturb = state->Amp_perturb * cos(2 * PI * x*cos(state->phi) / state->Lperiod_perturb+2 * PI * x*sin(state->phi) / state->Lperiod_perturb);
  double V_perturb = state->Amp_perturb * sin(2 * PI * y / state->Lperiod_perturb);
  return V+V_perturb;
}
