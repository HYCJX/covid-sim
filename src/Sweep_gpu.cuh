#ifndef COVIDSIM_SWEEP_GPU_CUH
#define COVIDSIM_SWEEP_GPU_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "CalcInfSusc.h"
#include "Dist.h"
#include "InfStat.h"
#include "Model.h"
#include "ModelMacros.h"
#include "Param.h"
#include "Rand.h"
#include "Update.h"

/* GPU Scheduling Constants */
#define imin(a, b) (a<b?a:b)
constexpr int min_infected = 100000;
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = imin(32, (min_infected + threadsPerBlock - 1) / threadsPerBlock);

/* CPU Variables Holding Inner-Layer GPU Pointers */
extern struct Place **Struct_Builder;
extern struct Place **Places_Builder;
extern int **SamplingQueue_Builder;
extern struct PopVar *StateT_Builder;
extern struct Data *h_data; // Host array data.
extern struct Data *data_Builder; // Host non-array data & Device array data.

/* Interface to Main */
void Alloc_GPU(struct Lock *&Inf_Locks_GPU, struct Lock *&Dct_Locks_GPU, struct Lock *&Rand_Locks_GPU,
               struct Cell *&CellLookup_GPU,
               struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU,
               struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU,
               struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU,
               struct Param *&P_GPU, struct Data *&data_GPU);

void Free_GPU(struct Lock *&Inf_Locks_GPU, struct Lock *&Dct_Locks_GPU, struct Lock *&Rand_Locks_GPU,
              struct Cell *&CellLookup_GPU,
              struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU,
              struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU,
              struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU,
              struct Param *&P_GPU, struct Data *&data_GPU);

void InfectSweep_GPU(double t, int run, struct Lock *&Inf_Locks_GPU, struct Lock *&Dct_Locks_GPU,
                     struct Lock *&Rand_Locks_GPU,
                     struct Cell *&CellLookup_GPU, struct Person *&Hosts_GPU,
                     struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU,
                     struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU,
                     int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU,
                     struct Data *&data_GPU);

#endif //COVIDSIM_SWEEP_GPU_CUH
