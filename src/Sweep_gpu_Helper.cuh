#ifndef COVIDSIM_SWEEP_GPU_HELPER_CUH
#define COVIDSIM_SWEEP_GPU_HELPER_CUH

#include "Models/Cell.h"
#include "Models/Household.h"
#include "Models/Microcell.h"
#include "Models/Person.h"
#include "Constants.h"
#include "InfStat.h"
#include "Model.h"
#include "Param.h"
#include "Rand.h"
#include "Sweep_gpu.cuh"

/* Infect sweep variables */
struct Data {
    int bm /* Movement restrictions in place */;
    double s5; // Total spatial infectiousness summed over all infectious people in cell.
    double seasonality, sbeta, hbeta;
    double fp; // False positive.
    unsigned short int ts;
    bool need_exit;
    int exit_num;
};

/* Computation Kernel */
__global__ void
kernel(double t, int tn, Cell *c, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU, Household *Households_GPU,
       Microcell *Mcells_GPU, Place **Places_GPU, AdminUnit *AdUnits_GPU, int **SamplingQueue, PopVar *StateT_GPU,
       Param *P_GPU, Data *data);

/* --- Migration of External CPU Macros & Functions --- */

/* ModelMacros */
#define HOST_TREATED_GPU(x) ((Hosts_GPU[x].treat_stop_time > ts) && (Hosts_GPU[x].treat_start_time <= ts))
#define PLACE_CLOSED_GPU(x, y) ((Places_GPU[x][y].close_start_time <= ts) && (Places_GPU[x][y].close_end_time > ts))

#define HOST_VACCED_GPU(x) (Hosts_GPU[x].vacc_start_time+P_GPU->usVaccTimeToEfficacy<=ts)
#define HOST_VACCED_SWITCH_GPU(x) (Hosts_GPU[x].vacc_start_time >= P_GPU->usVaccTimeEfficacySwitch)
#define HOST_QUARANTINED_GPU(x) ((HostsQuarantine_GPU[x].comply == 1) && (HostsQuarantine_GPU[x].start_time + P_GPU->usHQuarantineHouseDuration > ts) && (HostsQuarantine_GPU[x].start_time <= ts))
#define HOST_ISOLATED_GPU(x) ((Hosts_GPU[x].isolation_start_time + P_GPU->usCaseIsolationDelay <= ts) && (Hosts_GPU[x].isolation_start_time + P_GPU->usCaseIsolationDelay + P_GPU->usCaseIsolationDuration > ts))
#define HOST_ABSENT_GPU(x) ((Hosts_GPU[x].absent_start_time <= ts) && (Hosts_GPU[x].absent_stop_time > ts))

#define HOST_AGE_GROUP_GPU(x) (Hosts_GPU[x].age / AGE_GROUP_WIDTH)

/* Dist */
extern __device__ double sinx_GPU[DEGREES_PER_TURN + 1];
extern __device__ double cosx_GPU[DEGREES_PER_TURN + 1];
extern __device__ double asin2sqx_GPU[1001];
__device__ double periodic_xy_GPU(double x, double y, Param *P_GPU);
__device__ double dist2UTM_GPU(double x1, double y1, double x2, double y2, Param *P_GPU);
__device__ double dist2_raw_GPU(double ax, double ay, double bx, double by, Param *P_GPU);

/* Rand */
extern __device__ int32_t Xcg1_GPU[MAX_NUM_THREADS * CACHE_LINE_SIZE];
extern __device__ int32_t Xcg2_GPU[MAX_NUM_THREADS * CACHE_LINE_SIZE];
__device__ double ranf_mt_GPU(int tn);
__device__ int32_t ignbin_mt_GPU(int32_t n, double pp, int tn);
__device__ void SampleWithoutReplacement_GPU(int tn, int k, int n, int **SamplingQueue_GPU);

/* CalcinfSusc */
__device__ double
CalcHouseInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                 Household *Households_GPU, Param *P_GPU);
__device__ double
CalcPlaceInf_GPU(int j, int k, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                 Param *P_GPU);
__device__ double
CalcSpatialInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                   Param *P_GPU);
__device__ double CalcPersonInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, Param *P_GPU);
__device__ double
CalcHouseSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU, Microcell *Mcells_GPU,
                  Param *P_GPU);
__device__ double CalcPlaceSusc_GPU(int ai, int k, unsigned short int ts, int infector, int tn, Person *Hosts_GPU,
                                    PersonQuarantine *HostsQuarantine_GPU, Microcell *Mcells_GPU, Param *P_GPU);
__device__ double CalcSpatialSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU,
                                      PersonQuarantine *HostsQuarantine_GPU, Microcell *Mcells_GPU, Param *P_GPU);
__device__ double
CalcPersonSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU, Param *P_GPU);

#endif //COVIDSIM_SWEEP_GPU_HELPER_CUH
