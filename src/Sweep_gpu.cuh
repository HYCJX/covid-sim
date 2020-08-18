#ifndef COVIDSIM_SWEEP_GPU_CUH
#define COVIDSIM_SWEEP_GPU_CUH

extern bool need_exit;
extern int exit_num;

void InfectSweep_GPU(double t, int run);

#endif //COVIDSIM_SWEEP_GPU_CUH
