#ifndef COVIDSIM_TEST_KERNEL_CUH
#define COVIDSIM_TEST_KERNEL_CUH

#include <iostream>

#include "Models/Cell.h"
#include "Models/Person.h"
#include "Model.h"
#include "Dist.h"
#include "Param.h"
#include "Rand.h"

// Test Cell:
void test_Cell(Cell *c);

// Test Hosts:
void test_Hosts(int index, int place_index, int place_type_index);

// Test HostsQuarantine:
void test_HostsQuarantine(int index);

// Test Households:
void test_HouseHolds(int index);

// Test Mcells:
void test_Mcells(int index);

// Test Places:
void test_Places(int i, int j, int group_start_i, int group_size_i, int members_i);

// Test AdUnits:
void test_AdUnits(int index);

// Test SamplingQueue:
void test_SamplingQueue(int i, int j);

// Test StateT:
void test_StateT(int index, int n_queue_index, int cell_inf_index);

// Test P:
void test_P();

// Test Static Data:
void test_static_data();

// Integration Test:
void test_all(Cell *c);

// Integration CudaMalloc Test:
void test_all_malloc(struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU, struct Data *&data);

// Integration CudaMemcpy Test:
void test_all_memcopy(struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU);

// Integration CudaMemcpy Back Test:
void test_all_memcopy2Host(struct Person *&Hosts_GPU, struct PersonQuarantine *&HostsQuarantine_GPU, struct Household *&Households_GPU, struct Microcell *&Mcells_GPU, struct Place **&Places_GPU, struct AdminUnit *&AdUnits_GPU, int **&SamplingQueue_GPU, struct PopVar *&StateT_GPU, struct Param *&P_GPU);

#endif //COVIDSIM_TEST_KERNEL_CUH