#ifndef COVIDSIM_SWEEP_GPU_CUH
#define COVIDSIM_SWEEP_GPU_CUH

#include "../../../../../usr/include/c++/7/cmath"
#include "../../../../../usr/include/c++/7/cstdio"
#include "../../../../../usr/include/c++/7/cstdlib"

#include "CalcInfSusc.h"
#include "Dist.h"
#include "InfStat.h"
#include "Model.h"
#include "ModelMacros.h"
#include "Param.h"
#include "Rand.h"
#include "Update.h"

void InfectSweep_GPU(double t, int run);

#endif //COVIDSIM_SWEEP_GPU_CUH
