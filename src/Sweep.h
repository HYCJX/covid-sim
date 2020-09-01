#ifndef COVIDSIM_SWEEP_H_INCLUDED_
#define COVIDSIM_SWEEP_H_INCLUDED_

extern int turn;
extern double total_time;

void TravelReturnSweep(float);
void TravelDepartSweep(float);
void InfectSweep(float, int); //added int as argument to InfectSweep to record run number: ggilani - 15/10/14
void IncubRecoverySweep(float, int); //added int as argument to record run number: ggilani - 15/10/14
int TreatSweep(float);
//void HospitalSweep(float); //added hospital sweep function: ggilani - 10/11/14
void DigitalContactTracingSweep(float); // added function to update contact tracing number

#endif // COVIDSIM_SWEEP_H_INCLUDED_
