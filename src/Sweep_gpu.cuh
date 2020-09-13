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

void Alloc_GPU(struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU, struct Data *&data);
void Free_GPU(struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU, struct Data *&data);
void InfectSweep_GPU(double t, int run);

#endif //COVIDSIM_SWEEP_GPU_CUH
