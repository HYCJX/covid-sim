#ifndef COVIDSIM_CALCINFSUSC_H_
#define COVIDSIM_CALCINFSUSC_H_

float CalcHouseInf(int, unsigned short int);
float CalcPlaceInf(int, int, unsigned short int);
float CalcSpatialInf(int, unsigned short int);
float CalcPersonInf(int, unsigned short int);
float CalcHouseSusc(int, unsigned short int, int, int);
float CalcPlaceSusc(int, int, unsigned short int, int, int);
float CalcSpatialSusc(int, unsigned short int, int, int);
float CalcPersonSusc(int, unsigned short int, int, int);

#endif // COVIDSIM_CALCINFSUSC_
