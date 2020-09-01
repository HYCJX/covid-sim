#ifndef COVIDSIM_DIST_H_INCLUDED_
#define COVIDSIM_DIST_H_INCLUDED_

#include "Models/Person.h"
#include "Models/Cell.h"
#include "Models/Microcell.h"
#include "Constants.h"

extern float sinx[DEGREES_PER_TURN + 1], cosx[DEGREES_PER_TURN + 1], asin2sqx[1001];
float dist2UTM(float, float, float, float);
float dist2(Person*, Person*);
float dist2_cc(Cell*, Cell*);
float dist2_cc_min(Cell*, Cell*);
float dist2_mm(Microcell*, Microcell*);
float dist2_raw(float, float, float, float);

#endif // COVIDSIM_DIST_H_INCLUDED_
