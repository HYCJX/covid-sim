#include "Test_Kernel.cuh"
#include "Sweep_gpu_Helper.cuh"

// Error Handling.
void handle_error(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

/* ----- Test Cell ----- */

__global__ void kernel_test_Cell(Cell *c, int j) {
    printf("GPU: c->I = %d.\n", c->I);
    printf("GPU: c->infected[%d] = %d.\n", j, c->infected[j]);
}

void test_Cell(Cell *c, int j) {
    struct Cell *c_GPU;
    struct Cell *c_Builder = (Cell *) malloc(sizeof(Cell));
    c_Builder->I = c->I;
    handle_error(cudaMalloc((void **) &c_Builder->infected, c->I * sizeof(int)));
    handle_error(cudaMemcpy(c_Builder->infected, c->infected, c->I * sizeof(int), cudaMemcpyHostToDevice));
    handle_error(cudaMalloc((void **) &c_GPU, sizeof(struct Cell)));
    handle_error(cudaMemcpy(c_GPU, c_Builder, sizeof(struct Cell), cudaMemcpyHostToDevice));

    kernel_test_Cell<<<1,1>>>(c_GPU, j);
    handle_error(cudaDeviceSynchronize());

    handle_error(cudaFree(c_Builder->infected));
    handle_error(cudaFree(c_GPU));
    free(c_Builder);
}

/* ----- Test Hosts ----- */

__global__ void kernel_test_Hosts(Person *Hosts_GPU, int index, int place_index, int place_group_index) {
    if (place_index >= 0) {
        printf("GPU: Hosts[%d].PlaceLinks[%d] = %d.\n", index, place_index, (Hosts_GPU + index) -> PlaceLinks[place_index]);
    }
    if (place_group_index >= 0){
        printf("GPU: Hosts[%d].PlaceGroupLinks[%d] = %d.\n", index, place_group_index, (Hosts_GPU + index) -> PlaceGroupLinks[place_group_index]);
    }

}

void test_Hosts(int index, int place_index, int place_group_index) {
    struct Person *Hosts_GPU;
    handle_error(cudaMalloc((void **) &Hosts_GPU, P.PopSize * sizeof(struct Person)));
    handle_error(cudaMemcpy(Hosts_GPU, Hosts, P.PopSize * sizeof(struct Person), cudaMemcpyHostToDevice));
    kernel_test_Hosts<<<1,1>>>(Hosts_GPU, index, place_index, place_group_index);
    handle_error(cudaDeviceSynchronize());
    handle_error(cudaMemcpy(Hosts, Hosts_GPU, P.PopSize * sizeof(struct Person), cudaMemcpyDeviceToHost));
    handle_error(cudaFree(Hosts_GPU));
}

/* ----- Test HostsQuarantine ----- */

__global__ void kernel_test_HostsQuarantine(PersonQuarantine *HostsQuarantine_GPU, int index) {
    printf("GPU: HostsQuarantine[%d].comply = %d.\n", index, HostsQuarantine_GPU[index].comply);
    printf("GPU: HostsQuarantine[%d].start_time = %d.\n", index, HostsQuarantine_GPU[index].start_time);
}

void test_HostsQuarantine(int index) {
    struct PersonQuarantine *HostsQuarantine_GPU;
    handle_error(cudaMalloc((void **) &HostsQuarantine_GPU, HostsQuarantine.size() * sizeof(struct PersonQuarantine)));
    handle_error(cudaMemcpy(HostsQuarantine_GPU, &HostsQuarantine[0], HostsQuarantine.size() * sizeof(struct PersonQuarantine), cudaMemcpyHostToDevice));
    kernel_test_HostsQuarantine<<<1,1>>>(HostsQuarantine_GPU, index);
    handle_error(cudaDeviceSynchronize());
    handle_error(cudaFree(HostsQuarantine_GPU));
}

/* ----- Test Households ----- */

__global__ void kernel_test_Households(Household *Households_GPU, int index) {
    printf("GPU: Households[%d].FirstPerson = %d.\n", index, Households_GPU[index].FirstPerson);
    printf("GPU: Households[%d].nh = %u.\n", index, Households_GPU[index].nh);
    printf("GPU: Households[%d].nhr = %u.\n", index, Households_GPU[index].nhr);
    printf("GPU: Households[%d].loc.x = %9.6f.\n", index, Households_GPU[index].loc.x);
    printf("GPU: Households[%d].loc.y = %9.6f.\n", index, Households_GPU[index].loc.y);
}

void test_HouseHolds(int index) {
    struct Household *Households_GPU;
    handle_error(cudaMalloc((void **) &Households_GPU, P.NH * sizeof(struct Household)));
    handle_error(cudaMemcpy(Households_GPU, Households, P.NH * sizeof(struct Household), cudaMemcpyHostToDevice));
    kernel_test_Households<<<1,1>>>(Households_GPU, index);
    handle_error(cudaDeviceSynchronize());
    handle_error(cudaFree(Households_GPU));
}

/* ----- Test Mcells ----- */

__global__ void kernel_test_Mcells(Microcell *Mcells_GPU, int index) {
    printf("GPU: Mcells[%d].adunit = %d.\n", index, Mcells_GPU[index].adunit);
    printf("GPU: Mcells[%d].moverest = %u.\n", index, Mcells_GPU[index].moverest);
    printf("GPU: Mcells[%d].socdist = %u.\n", index, Mcells_GPU[index].socdist);
}

void test_Mcells(int index) {
    struct Microcell *Mcells_GPU;
    handle_error(cudaMalloc((void **) &Mcells_GPU, P.NMC * sizeof(struct Microcell)));
    handle_error(cudaMemcpy(Mcells_GPU, Mcells, P.NMC * sizeof(struct Microcell), cudaMemcpyHostToDevice));
    kernel_test_Mcells<<<1,1>>>(Mcells_GPU, index);
    handle_error(cudaDeviceSynchronize());
    handle_error(cudaFree(Mcells_GPU));
}

/* ----- Test Places -----*/

__global__ void kernel_test_places(Place **Places_GPU, int i, int j, int group_start_i, int group_size_i, int members_i) {
    printf("GPU: Places[%d][%d].n = %d.\n", i, j, Places_GPU[i][j].n);
    printf("GPU: Places[%d][%d].mcell = %d.\n", i, j, Places_GPU[i][j].mcell);
    printf("GPU: Places[%d][%d].loc.x = %9.6f.\n", i, j, Places_GPU[i][j].loc.x);
    printf("GPU: Places[%d][%d].loc.y = %9.6f.\n", i, j, Places_GPU[i][j].loc.y);
    if (group_start_i >= 0) {
        printf("GPU: Places[%d][%d].group_start[%d] = %d.\n", i, j, group_start_i, Places_GPU[i][j].group_start[group_start_i]);
    }
    if (group_size_i >= 0) {
        printf("GPU: Places[%d][%d].group_size[%d] = %d.\n", i, j, group_size_i, Places_GPU[i][j].group_size[group_size_i]);
    }
    if (members_i >= 0) {
        printf("GPU: Places[%d][%d].members[%d] = %d.\n", i, j, members_i, Places_GPU[i][j].members[members_i]);
    }
}

void test_Places(int i, int j, int group_start_i, int group_size_i, int members_i){
    /* --- Start Time Record --- */
    cudaEvent_t start, stop;
    handle_error(cudaEventCreate(&start));
    handle_error(cudaEventCreate(&stop));
    handle_error(cudaEventRecord(start, 0));
    /* ---                   --- */

    struct Place **Struct_Builder = (struct Place **) malloc(P.PlaceTypeNum * sizeof(struct Place *));
    for (int p = 0; p < P.PlaceTypeNum; p++) {
        Struct_Builder[p] = (struct Place *) malloc(P.Nplace[p] * sizeof(struct Place));
        for (int q = 0; q < P.Nplace[p]; q++) {
            Place place = Places[p][q];
            Struct_Builder[p][q] = place;
            handle_error(cudaMalloc((void **) &Struct_Builder[p][q].group_start, place.ng * sizeof(int)));
            handle_error(cudaMemcpy(Struct_Builder[p][q].group_start, place.group_start, place.ng * sizeof(int), cudaMemcpyHostToDevice));
            handle_error(cudaMalloc((void **) &Struct_Builder[p][q].group_size, place.ng * sizeof(int)));
            handle_error(cudaMemcpy(Struct_Builder[p][q].group_size, place.group_size, place.ng * sizeof(int), cudaMemcpyHostToDevice));
            if (p == P.HotelPlaceType) {
                handle_error(cudaMalloc((void **) &Struct_Builder[p][q].members, 2 * ((int)P.PlaceTypeMeanSize[p]) * sizeof(int)));
                handle_error(cudaMemcpy(Struct_Builder[p][q].members, place.members, 2 * ((int)P.PlaceTypeMeanSize[p]) * sizeof(int), cudaMemcpyHostToDevice));
            } else {
                handle_error(cudaMalloc((void **) &Struct_Builder[p][q].members, place.n * sizeof(int)));
                handle_error(cudaMemcpy(Struct_Builder[p][q].members, place.members, place.n * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }
    struct Place **Places_GPU;
    struct Place *Places_Builder[P.PlaceTypeNum];
    handle_error(cudaMalloc((void **) &Places_GPU, P.PlaceTypeNum * sizeof(struct Place *)));
    for (int m = 0; m < P.PlaceTypeNum; m++) {
        handle_error(cudaMalloc((void **) &Places_Builder[m], P.Nplace[m] * sizeof(struct Place)));
        handle_error(cudaMemcpy(Places_Builder[m], Struct_Builder[m], P.Nplace[m] * sizeof(struct Place), cudaMemcpyHostToDevice));
    }
    handle_error(cudaMemcpy(Places_GPU, Places_Builder, P.PlaceTypeNum * sizeof(struct Place *),cudaMemcpyHostToDevice));

//    kernel_test_places<<<1,1>>>(Places_GPU, i, j, group_start_i, group_size_i, members_i);
//    handle_error(cudaDeviceSynchronize());

    for (int m = 0; m < P.PlaceTypeNum; m++) {
        for (int n = 0; n < P.Nplace[m]; n++) {
            handle_error(cudaFree(Struct_Builder[m][n].group_start));
            handle_error(cudaFree(Struct_Builder[m][n].group_size));
            handle_error(cudaFree(Struct_Builder[m][n].members));
        }
        free(Struct_Builder[m]);
    }
    free(Struct_Builder);
    for (int i = 0; i < P.PlaceTypeNum; i++) {
        handle_error(cudaFree(Places_Builder[i]));
    }
    handle_error(cudaFree(Places_GPU));
    /* --- Stop Time Record --- */
    handle_error(cudaEventRecord(stop, 0));
    handle_error(cudaEventSynchronize(stop));
    float elapsedTime;
    handle_error(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Data Transfer Time: %3.lf ms.\n", elapsedTime);
    handle_error(cudaEventDestroy(start));
    handle_error(cudaEventDestroy(stop));
    /* ---                  --- */
}

/* ----- Test AdUnits ----- */

__global__ void kernel_test_AdUnits(AdminUnit *AdUnits_GPU, int index) {
    printf("GPU: AdUnits[%d].n = %d.\n", index, AdUnits_GPU[index].n);
    printf("GPU: AdUnits[%d].DigitalContactTracingTimeStart = %lf.\n", index, AdUnits_GPU[index].DigitalContactTracingTimeStart);
}

void test_AdUnits(int index) {
    struct AdminUnit *AdUnits_GPU;
    handle_error(cudaMalloc((void **) &AdUnits_GPU, MAX_ADUNITS * sizeof(struct AdminUnit)));
    handle_error(cudaMemcpy(AdUnits_GPU, AdUnits, MAX_ADUNITS * sizeof(struct AdminUnit), cudaMemcpyHostToDevice));

    kernel_test_AdUnits<<<1,1>>>(AdUnits_GPU, index);

    handle_error(cudaFree(AdUnits_GPU));
}

/* ----- Test SamplingQueue ----- */

__global__ void kernel_test_SamplingQueue(int **SamplingQueue_GPU, int i, int j) {
    printf("GPU: SamplingQueue[%d][%d] = %d.\n", i, j, SamplingQueue_GPU[i][j]);
}

void test_SamplingQueue(int i, int j) {
    int **SamplingQueue_GPU;
    int *SamplingQueue_Builder[P.NumThreads];
    handle_error(cudaMalloc((void **) &SamplingQueue_GPU, P.NumThreads * sizeof(int *)));
    for (int i = 0; i < P.NumThreads; i++) {
        handle_error(cudaMalloc((void **) &SamplingQueue_Builder[i],2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int)));
        handle_error(cudaMemcpy(SamplingQueue_Builder[i], SamplingQueue[i],2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int), cudaMemcpyHostToDevice));
    }
    handle_error(cudaMemcpy(SamplingQueue_GPU, SamplingQueue_Builder, P.NumThreads * sizeof(int *),cudaMemcpyHostToDevice));

    kernel_test_SamplingQueue<<<1,1>>>(SamplingQueue_GPU, i, j);
    handle_error(cudaDeviceSynchronize());

    handle_error(cudaMemcpy(SamplingQueue_Builder, SamplingQueue_GPU, P.NumThreads * sizeof(int *),cudaMemcpyDeviceToHost));
    for (int i = 0; i < P.NumThreads; i++) {
        handle_error(cudaMemcpy(SamplingQueue[i], SamplingQueue_Builder[i], 2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
        handle_error(cudaFree(SamplingQueue_Builder[i]));
    }
    cudaFree(SamplingQueue_GPU);
}

/* ----- Test StateT ----- */

__global__ void kernel_test_StateT(PopVar *StateT_GPU, int index, int n_queue_index, int cell_inf_index) {
    if (n_queue_index >= 0) {
        printf("GPU: StateT[%d].n_queue[%d] = %d.\n", index, n_queue_index, StateT_GPU[index].n_queue[n_queue_index]);
    }
    if (cell_inf_index >= 0) {
        printf("GPU: StateT[%d].cell_inf[%d] = %9.6f.\n", index, cell_inf_index, StateT_GPU[index].cell_inf[cell_inf_index]);
    }
    StateT_GPU[index].n_queue[n_queue_index] *= 10;
    StateT_GPU[index].cell_inf[cell_inf_index] *= 10;
}

void test_StateT(int index, int n_queue_index, int cell_inf_index) {
    struct PopVar *StateT_GPU;
    struct PopVar *StateT_Builder = (struct PopVar *) malloc(P.NumThreads * sizeof(struct PopVar));
    memcpy(StateT_Builder, StateT, P.NumThreads * sizeof(struct PopVar));
    for (int i = 0; i < P.NumThreads; i++) {
        for (int j = 0; j < P.NumThreads; j++) {
            handle_error(cudaMalloc((void **) &(StateT_Builder[i].inf_queue[j]),StateT[i].n_queue[j] * sizeof(Infection)));
            handle_error(cudaMemcpy(StateT_Builder[i].inf_queue[j], StateT[i].inf_queue[j],StateT[i].n_queue[j] * sizeof(Infection),cudaMemcpyHostToDevice));
        }
        handle_error(cudaMalloc((void **) &StateT_Builder[i].cell_inf, StateT[i].cell_inf_length * sizeof(float)));
        handle_error(cudaMemcpy(StateT_Builder[i].cell_inf, StateT[i].cell_inf, StateT[i].cell_inf_length * sizeof(float),cudaMemcpyHostToDevice));
        for (int j = 0; j < P.NumAdunits; j++) {
            handle_error(cudaMalloc((void **) &(StateT_Builder[i].dct_queue[j]), StateT[i].ndct_queue[j] * sizeof(ContactEvent)));
            handle_error(cudaMemcpy(StateT_Builder[i].dct_queue[j], StateT[i].dct_queue[j], StateT[i].ndct_queue[j] * sizeof(ContactEvent), cudaMemcpyHostToDevice));
        }
    }
    handle_error(cudaMalloc((void **) &StateT_GPU, P.NumThreads * sizeof(struct PopVar)));
    handle_error(cudaMemcpy(StateT_GPU, StateT_Builder, P.NumThreads * sizeof(struct PopVar),cudaMemcpyHostToDevice));

    kernel_test_StateT<<<1,1>>>(StateT_GPU, index, n_queue_index, cell_inf_index);
    handle_error(cudaDeviceSynchronize());

    handle_error(cudaMemcpy(StateT_Builder, StateT_GPU, P.NumThreads * sizeof(struct PopVar),cudaMemcpyDeviceToHost));
    handle_error(cudaFree(StateT_GPU));
    for (int i = 0; i < P.NumThreads; i++) {
        for (int j = 0; j < MAX_NUM_THREADS; j++) {
            handle_error(cudaMemcpy(StateT[i].inf_queue[j], StateT_Builder[i].inf_queue[j], StateT[i].n_queue[j] * sizeof(Infection), cudaMemcpyDeviceToHost));
            handle_error(cudaFree(StateT_Builder[i].inf_queue[j]));
        }
        memcpy(StateT[i].n_queue, StateT_Builder[i].n_queue, MAX_NUM_THREADS * sizeof(int));
        handle_error(cudaFree(StateT_Builder[i].cell_inf));
        for (int j = 0; j < P.NumAdunits; j++) {
            handle_error(cudaMemcpy(StateT[i].dct_queue[j], StateT_Builder[i].dct_queue[j], StateT[i].ndct_queue[j] * sizeof(ContactEvent), cudaMemcpyDeviceToHost));
            handle_error(cudaFree(StateT_Builder[i].dct_queue[j]));
        }
    }
    free(StateT_Builder);
}

/*Test Static Data */

__global__ void kernel_test_static_data() {
    printf("GPU: sinx_GPU[%d] = %f.\n", 1, sinx_GPU[1]);
    printf("GPU: cosx_GPU[%d] = %f.\n", 1, cosx_GPU[1]);
    printf("GPU: asin2sqx_GPU[%d] = %f.\n", 1, asin2sqx_GPU[1]);
    printf("GPU: Xcg1_GPU[%d] = %d.\n", 1, Xcg1_GPU[1]);
    printf("GPU: Xcg2_GPU[%d] = %d.\n", 1, Xcg2_GPU[1]);
}

void test_static_data() {
//    handle_error(cudaMemcpyToSymbol(sinx_GPU, sinx, (DEGREES_PER_TURN + 1) * sizeof(double)));
//    handle_error(cudaMemcpyToSymbol(cosx_GPU, cosx, (DEGREES_PER_TURN + 1) * sizeof(double)));
//    handle_error(cudaMemcpyToSymbol(asin2sqx_GPU, asin2sqx, (1001) * sizeof(double)));
//    handle_error(cudaMemcpyToSymbol(Xcg1_GPU, Xcg1, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
//    handle_error(cudaMemcpyToSymbol(Xcg2_GPU, Xcg2, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
    kernel_test_static_data<<<1,1>>>();
    handle_error(cudaDeviceSynchronize());
}

/* Test P */

__global__ void kernel_test_P() {}

void test_P() {
    struct Param *P_GPU;
    handle_error(cudaMalloc((void **) &P_GPU, sizeof(struct Param)));
    handle_error(cudaMemcpy(P_GPU, &P, sizeof(struct Param), cudaMemcpyHostToDevice));
    handle_error(cudaMemcpy(&P, P_GPU, sizeof(struct Param), cudaMemcpyDeviceToHost));
    handle_error(cudaFree(P_GPU));
}

/* Integration Test */

void test_all(Cell *c) {

    /* --- Start Time Record --- */
    cudaEvent_t start, stop;
    handle_error(cudaEventCreate(&start));
    handle_error(cudaEventCreate(&stop));
    handle_error(cudaEventRecord(start, 0));
    /* ---                   --- */

    /* --- Copy Data: Host to Device --- */
    // Dist global variables:
    handle_error(cudaMemcpyToSymbol(sinx_GPU, sinx, (DEGREES_PER_TURN + 1) * sizeof(double)));
    handle_error(cudaMemcpyToSymbol(cosx_GPU, cosx, (DEGREES_PER_TURN + 1) * sizeof(double)));
    // Rand global variables:
    handle_error(cudaMemcpyToSymbol(asin2sqx_GPU, asin2sqx, (1001) * sizeof(double)));
    handle_error(cudaMemcpyToSymbol(Xcg1_GPU, Xcg1, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
    handle_error(cudaMemcpyToSymbol(Xcg2_GPU, Xcg2, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
    // Cell:
    struct Cell *c_GPU;
    struct Cell *c_Builder = (Cell *) malloc(sizeof(Cell));
    c_Builder->I = c->I;
    handle_error(cudaMalloc((void **) &c_Builder->infected, c->I * sizeof(int)));
    handle_error(cudaMemcpy(c_Builder->infected, c->infected, c->I * sizeof(int), cudaMemcpyHostToDevice));
    handle_error(cudaMalloc((void **) &c_GPU, sizeof(struct Cell)));
    handle_error(cudaMemcpy(c_GPU, c_Builder, sizeof(struct Cell), cudaMemcpyHostToDevice));
    // Hosts:
    struct Person *Hosts_GPU;
    handle_error(cudaMalloc((void **) &Hosts_GPU, P.PopSize * sizeof(struct Person)));
    handle_error(cudaMemcpy(Hosts_GPU, Hosts, P.PopSize * sizeof(struct Person), cudaMemcpyHostToDevice));
    // HostsQuarantine:
    struct PersonQuarantine *HostsQuarantine_GPU;
    handle_error(cudaMalloc((void **) &HostsQuarantine_GPU, HostsQuarantine.size() * sizeof(struct PersonQuarantine)));
    handle_error(cudaMemcpy(HostsQuarantine_GPU, &HostsQuarantine[0], HostsQuarantine.size() * sizeof(struct PersonQuarantine), cudaMemcpyHostToDevice));
    // Households:
    struct Household *Households_GPU;
    handle_error(cudaMalloc((void **) &Households_GPU, P.NH * sizeof(struct Household)));
    handle_error(cudaMemcpy(Households_GPU, Households, P.NH * sizeof(struct Household), cudaMemcpyHostToDevice));
    // Mcells:
    struct Microcell *Mcells_GPU;
    handle_error(cudaMalloc((void **) &Mcells_GPU, P.NMC * sizeof(struct Microcell)));
    handle_error(cudaMemcpy(Mcells_GPU, Mcells, P.NMC * sizeof(struct Microcell), cudaMemcpyHostToDevice));
    // Places:
    struct Place **Struct_Builder = (struct Place **) malloc(P.PlaceTypeNum * sizeof(struct Place *));
    for (int p = 0; p < P.PlaceTypeNum; p++) {
        Struct_Builder[p] = (struct Place *) malloc(P.Nplace[p] * sizeof(struct Place));
        for (int q = 0; q < P.Nplace[p]; q++) {
            Place place = Places[p][q];
            Struct_Builder[p][q] = place;
            handle_error(cudaMalloc((void **) &Struct_Builder[p][q].group_start, place.ng * sizeof(int)));
            handle_error(cudaMemcpy(Struct_Builder[p][q].group_start, place.group_start, place.ng * sizeof(int), cudaMemcpyHostToDevice));
            handle_error(cudaMalloc((void **) &Struct_Builder[p][q].group_size, place.ng * sizeof(int)));
            handle_error(cudaMemcpy(Struct_Builder[p][q].group_size, place.group_size, place.ng * sizeof(int), cudaMemcpyHostToDevice));
            if (p == P.HotelPlaceType) {
                handle_error(cudaMalloc((void **) &Struct_Builder[p][q].members, 2 * ((int)P.PlaceTypeMeanSize[p]) * sizeof(int)));
                handle_error(cudaMemcpy(Struct_Builder[p][q].members, place.members, 2 * ((int)P.PlaceTypeMeanSize[p]) * sizeof(int), cudaMemcpyHostToDevice));
            } else {
                handle_error(cudaMalloc((void **) &Struct_Builder[p][q].members, place.n * sizeof(int)));
                handle_error(cudaMemcpy(Struct_Builder[p][q].members, place.members, place.n * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }
    struct Place **Places_GPU;
    struct Place *Places_Builder[P.PlaceTypeNum];
    handle_error(cudaMalloc((void **) &Places_GPU, P.PlaceTypeNum * sizeof(struct Place *)));
    for (int m = 0; m < P.PlaceTypeNum; m++) {
        handle_error(cudaMalloc((void **) &Places_Builder[m], P.Nplace[m] * sizeof(struct Place)));
        handle_error(cudaMemcpy(Places_Builder[m], Struct_Builder[m], P.Nplace[m] * sizeof(struct Place), cudaMemcpyHostToDevice));
    }
    handle_error(cudaMemcpy(Places_GPU, Places_Builder, P.PlaceTypeNum * sizeof(struct Place *),cudaMemcpyHostToDevice));

    /* ---                           --- */

    /* --- Copy Data: Device to Host & Free Memory --- */
    // Cell:
    handle_error(cudaFree(c_Builder->infected));
    handle_error(cudaFree(c_GPU));
    free(c_Builder);
    // Hosts:
    handle_error(cudaMemcpy(Hosts, Hosts_GPU, P.PopSize * sizeof(struct Person), cudaMemcpyDeviceToHost));
    handle_error(cudaFree(Hosts_GPU));
    // HostsQuarantine:
    handle_error(cudaFree(HostsQuarantine_GPU));
    // Households:
    handle_error(cudaFree(Households_GPU));
    // Mcells:
    handle_error(cudaFree(Mcells_GPU));
    // Places:
    for (int m = 0; m < P.PlaceTypeNum; m++) {
        for (int n = 0; n < P.Nplace[m]; n++) {
            handle_error(cudaFree(Struct_Builder[m][n].group_start));
            handle_error(cudaFree(Struct_Builder[m][n].group_size));
            handle_error(cudaFree(Struct_Builder[m][n].members));
        }
        free(Struct_Builder[m]);
    }
    free(Struct_Builder);
    for (int i = 0; i < P.PlaceTypeNum; i++) {
        handle_error(cudaFree(Places_Builder[i]));
    }
    handle_error(cudaFree(Places_GPU));

    /* ---                                         --- */

    /* --- Stop Time Record --- */
    handle_error(cudaEventRecord(stop, 0));
    handle_error(cudaEventSynchronize(stop));
    float elapsedTime;
    handle_error(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Data Transfer Time: %3.lf ms.\n", elapsedTime);
    handle_error(cudaEventDestroy(start));
    handle_error(cudaEventDestroy(stop));
    /* ---                  --- */
}