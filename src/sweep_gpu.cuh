//
// Created by gpu on 02/08/2020.
//

#ifndef COVIDSIM_SWEEP_GPU_CUH
#define COVIDSIM_SWEEP_GPU_CUH

void InfectSweep_GPU(double t, int run);
void Record_Time(void (*f)(double, int), double p1, int p2);

#endif //COVIDSIM_SWEEP_GPU_CUH
