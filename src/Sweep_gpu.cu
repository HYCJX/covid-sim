#include "Sweep_gpu.cuh"
#include "Sweep_gpu_Helper.cuh"

/* Helpers for CUDA */
void HANDLE_ERROR(cudaError_t error);

void InfectSweep_GPU(double t, int run) {

    int n; // Number of people you could potentially infect in your place group, then number of potential spatial infections doled out by cell on other cells.
    int f, f2, cq /* Cell queue */, bm /* Movement restrictions in place */, ci /* Person index */;
    double s; // Household Force Of Infection (FOI) on fellow household member, then place susceptibility, then random number for spatial infections allocation.
    double s2; // Spatial infectiousness, then distance in spatial infections allocation.
    double s3, s3_scaled; // Household, then place infectiousness.
    double s4, s4_scaled; // Place infectiousness (copy of s3 as some code commented out.
    double s5; // Total spatial infectiousness summed over all infectious people in cell.
    double s6;
    double seasonality, sbeta, hbeta;
    double fp; // False positive.
    unsigned short int ts;

    // If not doing seasonality:
    if (!P.DoSeasonality) {
        // Set seasonality to 1.
        seasonality = 1.0;
    } else {
        // Otherwise pick seasonality from P.Seasonality array using day number in year.
        seasonality = P.Seasonality[((int) t) % DAYS_PER_YEAR];
    }
    // ts = the timestep number of the start of the current day
    ts = (unsigned short int) (P.TimeStepsPerDay * t);
    // fp = false positive
    fp = P.TimeStep / (1 - P.FalsePositiveRate);
    // sbeta seasonality beta
    sbeta = seasonality * fp * P.LocalBeta;
    // hbeta = household beta
    // if doing households, hbeta = seasonality * fp * P.HouseholdTrans, else hbeta = 0
    hbeta = (P.DoHouseholds) ? (seasonality * fp * P.HouseholdTrans) : 0;
    // Establish if movement restrictions are in place on current day - store in bm, 0:false, 1:true
    bm = ((P.DoBlanketMoveRestr) && (t >= P.MoveRestrTimeStart) && (t < P.MoveRestrTimeStart + P.MoveRestrDuration));
    // File for storing error reports
    FILE *stderr_shared = stderr;

#pragma omp parallel for private(n, f, f2, s, s2, s3, s4, s5, s6, cq, ci, s3_scaled, s4_scaled) schedule(static, 1) default(none) \
        shared(t, P, CellLookup, Hosts, AdUnits, Households, Places, SamplingQueue, Cells, Mcells, StateT, hbeta, sbeta, seasonality, ts, fp, bm, stderr_shared)
    for (int tn = 0; tn < P.NumThreads; tn++)
        for (int b = tn; b < P.NCP; b += P.NumThreads) // Loop over (in parallel) all populated cells.
        {
            Cell *c = CellLookup[b]; // Select Cell given by index b.
            s5 = 0; // Spatial infectiousness summed over all infectious people in loop below.

            /* --- Copy Data: Host to Device --- */
//            // Dist global variables:
//            HANDLE_ERROR(cudaMemcpyToSymbol(sinx_GPU, sinx, (DEGREES_PER_TURN + 1) * sizeof(double)));
//            HANDLE_ERROR(cudaMemcpyToSymbol(cosx_GPU, cosx, (DEGREES_PER_TURN + 1) * sizeof(double)));
//            HANDLE_ERROR(cudaMemcpyToSymbol(asin2sqx_GPU, asin2sqx, (1001) * sizeof(double)));
//            // Rand global variables:
//            HANDLE_ERROR(cudaMemcpyToSymbol(Xcg1_GPU, Xcg1, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
//            HANDLE_ERROR(cudaMemcpyToSymbol(Xcg2_GPU, Xcg2, (MAX_NUM_THREADS * CACHE_LINE_SIZE) * sizeof(int32_t)));
            // Cell:
            struct Cell *c_GPU;
            struct Cell *c_Builder = (Cell *) malloc(sizeof(Cell));
            c_Builder->I = c->I;
            HANDLE_ERROR(cudaMalloc((void **) &c_Builder->infected, c->I * sizeof(int)));
            HANDLE_ERROR(cudaMemcpy(c_Builder->infected, c->infected, c->I * sizeof(int), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMalloc((void **) &c_GPU, sizeof(struct Cell)));
            HANDLE_ERROR(cudaMemcpy(c_GPU, c_Builder, sizeof(struct Cell), cudaMemcpyHostToDevice));
            // Hosts:
            struct Person *Hosts_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &Hosts_GPU, P.PopSize * sizeof(struct Person)));
            HANDLE_ERROR(cudaMemcpy(Hosts_GPU, Hosts, P.PopSize * sizeof(struct Person), cudaMemcpyHostToDevice));
            // HostsQuarantine:
            struct PersonQuarantine *HostsQuarantine_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &HostsQuarantine_GPU, HostsQuarantine.size() * sizeof(struct PersonQuarantine)));
            HANDLE_ERROR(cudaMemcpy(HostsQuarantine_GPU, &HostsQuarantine[0], HostsQuarantine.size() * sizeof(struct PersonQuarantine), cudaMemcpyHostToDevice));
            // Households:
            struct Household *Households_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &Households_GPU, P.NH * sizeof(struct Household)));
            HANDLE_ERROR(cudaMemcpy(Households_GPU, Households, P.NH * sizeof(struct Household), cudaMemcpyHostToDevice));
            // Mcells:
            struct Microcell *Mcells_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &Mcells_GPU, P.NMC * sizeof(struct Microcell)));
            HANDLE_ERROR(cudaMemcpy(Mcells_GPU, Mcells, P.NMC * sizeof(struct Microcell), cudaMemcpyHostToDevice));
            // Places:
            struct Place **Struct_Builder = (struct Place **) malloc(P.PlaceTypeNum * sizeof(struct Place *));
            for (int i = 0; i < P.PlaceTypeNum; i++) {
                Struct_Builder[i] = (struct Place *) malloc(P.Nplace[i] * sizeof(struct Place));
                for (int j = 0; j < P.Nplace[i]; j++) {
                    Place place = Places[i][j];
                    Struct_Builder[i][j] = place;
                    HANDLE_ERROR(cudaMalloc((void **) &Struct_Builder[i][j].group_start, place.ng * sizeof(int)));
                    HANDLE_ERROR(cudaMemcpy(Struct_Builder[i][j].group_start, place.group_start, place.ng * sizeof(int), cudaMemcpyHostToDevice));
                    HANDLE_ERROR(cudaMalloc((void **) &Struct_Builder[i][j].group_size, place.ng * sizeof(int)));
                    HANDLE_ERROR(cudaMemcpy(Struct_Builder[i][j].group_size, place.group_size, place.ng * sizeof(int), cudaMemcpyHostToDevice));
                    if (i == P.HotelPlaceType) {
                        HANDLE_ERROR(cudaMalloc((void **) &Struct_Builder[i][j].members, 2 * ((int)P.PlaceTypeMeanSize[i]) * sizeof(int)));
                        HANDLE_ERROR(cudaMemcpy(Struct_Builder[i][j].members, place.members, 2 * ((int)P.PlaceTypeMeanSize[i]) * sizeof(int), cudaMemcpyHostToDevice));
                    } else {
                        HANDLE_ERROR(cudaMalloc((void **) &Struct_Builder[i][j].members, place.n * sizeof(int)));
                        HANDLE_ERROR(cudaMemcpy(Struct_Builder[i][j].members, place.members, place.n * sizeof(int), cudaMemcpyHostToDevice));
                    }
                }
            }
            struct Place **Places_GPU;
            struct Place *Places_Builder[P.PlaceTypeNum];
            HANDLE_ERROR(cudaMalloc((void **) &Places_GPU, P.PlaceTypeNum * sizeof(struct Place *)));
            for (int i = 0; i < P.PlaceTypeNum; i++) {
                HANDLE_ERROR(cudaMalloc((void **) &Places_Builder[i], P.Nplace[i] * sizeof(struct Place)));
                HANDLE_ERROR(cudaMemcpy(Places_Builder[i], Struct_Builder[i], P.Nplace[i] * sizeof(struct Place), cudaMemcpyHostToDevice));
            }
            HANDLE_ERROR(cudaMemcpy(Places_GPU, Places_Builder, P.PlaceTypeNum * sizeof(struct Place *), cudaMemcpyHostToDevice));
            // AdUnits:
            struct AdminUnit *AdUnits_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &AdUnits_GPU, MAX_ADUNITS * sizeof(struct AdminUnit)));
            HANDLE_ERROR(cudaMemcpy(AdUnits_GPU, AdUnits, MAX_ADUNITS * sizeof(struct AdminUnit), cudaMemcpyHostToDevice));
            // SamplingQueue:
            int **SamplingQueue_GPU;
            int *SamplingQueue_Builder[P.NumThreads];
            HANDLE_ERROR(cudaMalloc((void **) &SamplingQueue_GPU, P.NumThreads * sizeof(int *)));
            for (int i = 0; i < P.NumThreads; i++) {
                HANDLE_ERROR(cudaMalloc((void **) &SamplingQueue_Builder[i], 2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int)));
                HANDLE_ERROR(cudaMemcpy(SamplingQueue_Builder[i], SamplingQueue[i], 2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int), cudaMemcpyHostToDevice));
            }
            HANDLE_ERROR(cudaMemcpy(SamplingQueue_GPU, SamplingQueue_Builder, P.NumThreads * sizeof(int *), cudaMemcpyHostToDevice));
            // StateT:
            struct PopVar *StateT_GPU;
            struct PopVar *StateT_Builder = (struct PopVar *) malloc(P.NumThreads * sizeof(struct PopVar));
            memcpy(StateT_Builder, StateT, P.NumThreads * sizeof(struct PopVar));
            for (int i = 0; i < P.NumThreads; i++) {
                for (int j = 0; j < P.NumThreads; j++) {
                    HANDLE_ERROR(cudaMalloc((void **) &(StateT_Builder[i].inf_queue[j]),StateT[i].n_queue[j] * sizeof(Infection)));
                    HANDLE_ERROR(cudaMemcpy(StateT_Builder[i].inf_queue[j], StateT[i].inf_queue[j],StateT[i].n_queue[j] * sizeof(Infection),cudaMemcpyHostToDevice));
                }
                HANDLE_ERROR(cudaMalloc((void **) &StateT_Builder[i].cell_inf, StateT[i].cell_inf_length * sizeof(float)));
                HANDLE_ERROR(cudaMemcpy(StateT_Builder[i].cell_inf, StateT[i].cell_inf, StateT[i].cell_inf_length * sizeof(float),cudaMemcpyHostToDevice));
                for (int j = 0; j < P.NumAdunits; j++) {
                    HANDLE_ERROR(cudaMalloc((void **) &(StateT_Builder[i].dct_queue[j]), StateT[i].ndct_queue[j] * sizeof(ContactEvent)));
                    HANDLE_ERROR(cudaMemcpy(StateT_Builder[i].dct_queue[j], StateT[i].dct_queue[j], StateT[i].ndct_queue[j] * sizeof(ContactEvent), cudaMemcpyHostToDevice));
                }
            }
            HANDLE_ERROR(cudaMalloc((void **) &StateT_GPU, P.NumThreads * sizeof(struct PopVar)));
            HANDLE_ERROR(cudaMemcpy(StateT_GPU, StateT_Builder, P.NumThreads * sizeof(struct PopVar),cudaMemcpyHostToDevice));
            // P:
            struct Param *P_GPU;
            HANDLE_ERROR(cudaMalloc((void **) &P_GPU, sizeof(struct Param)));
            HANDLE_ERROR(cudaMemcpy(P_GPU, &P, sizeof(struct Param), cudaMemcpyHostToDevice));
            // Data:
            struct Data *data;
            HANDLE_ERROR(cudaMalloc((void **) &data, sizeof(struct Data)));
            struct Data *h_data = (struct Data *) malloc(sizeof(struct Data));
            h_data->bm = bm;
            h_data->s5 = s5;
            h_data->seasonality = seasonality;
            h_data->sbeta = sbeta;
            h_data->hbeta = hbeta;
            h_data->fp = fp;
            h_data->ts = ts;
            h_data->need_exit = false;
            h_data->exit_num = 0;
            HANDLE_ERROR(cudaMemcpy(data, h_data, sizeof(struct Data), cudaMemcpyHostToDevice));
            /* ---                           --- */

            /* --- Start Time Record --- */
            cudaEvent_t start, stop;
            HANDLE_ERROR(cudaEventCreate(&start));
            HANDLE_ERROR(cudaEventCreate(&stop));
            HANDLE_ERROR(cudaEventRecord(start, 0));
            /* ---                   --- */

            kernel<<<1, 1>>>(t, tn, c_GPU, Hosts_GPU, HostsQuarantine_GPU, Households_GPU, Mcells_GPU, Places_GPU, AdUnits_GPU, SamplingQueue, StateT_GPU, P_GPU, data);

            /* --- Stop Time Record --- */
            HANDLE_ERROR(cudaEventRecord(stop, 0));
            HANDLE_ERROR(cudaEventSynchronize(stop));
            float elapsedTime;
            HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
            printf("Infect Sweep Time: %3.lf ms.\n", elapsedTime);
            HANDLE_ERROR(cudaEventDestroy(start));
            HANDLE_ERROR(cudaEventDestroy(stop));
            /* ---                  --- */

            /* --- Copy Data: Device to Host & Free Memory --- */
            // Cell:
            HANDLE_ERROR(cudaFree(c_Builder->infected));
            HANDLE_ERROR(cudaFree(c_GPU));
            free(c_Builder);
            // Hosts:
            HANDLE_ERROR(cudaMemcpy(Hosts, Hosts_GPU, P.PopSize * sizeof(struct Person), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(Hosts_GPU));
            // HostsQuarantine:
            HANDLE_ERROR(cudaFree(HostsQuarantine_GPU));
            // Households:
            HANDLE_ERROR(cudaFree(Households_GPU));
            // Mcells:
            HANDLE_ERROR(cudaFree(Mcells_GPU));
            // Places:
            for (int i = 0; i < P.PlaceTypeNum; i++) {
                for (int j = 0; j < P.Nplace[i]; n++) {
                    HANDLE_ERROR(cudaFree(Struct_Builder[i][j].group_start));
                    HANDLE_ERROR(cudaFree(Struct_Builder[i][j].group_size));
                    HANDLE_ERROR(cudaFree(Struct_Builder[i][j].members));
                }
                free(Struct_Builder[i]);
            }
            free(Struct_Builder);
            for (int i = 0; i < P.PlaceTypeNum; i++) {
                HANDLE_ERROR(cudaFree(Places_Builder[i]));
            }
            HANDLE_ERROR(cudaFree(Places_GPU));
            // AdUnits:
            HANDLE_ERROR(cudaFree(AdUnits_GPU));
            // SamplingQueue:
            HANDLE_ERROR(cudaMemcpy(SamplingQueue_Builder, SamplingQueue_GPU, P.NumThreads * sizeof(int *),cudaMemcpyDeviceToHost));
            for (int i = 0; i < P.NumThreads; i++) {
                HANDLE_ERROR(cudaMemcpy(SamplingQueue[i], SamplingQueue_Builder[i], 2 * (MAX_PLACE_SIZE + CACHE_LINE_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaFree(SamplingQueue_Builder[i]));
            }
            cudaFree(SamplingQueue_GPU);
            // StateT:
            HANDLE_ERROR(cudaMemcpy(StateT_Builder, StateT_GPU, P.NumThreads * sizeof(struct PopVar),cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(StateT_GPU));
            for (int i = 0; i < P.NumThreads; i++) {
                for (int j = 0; j < MAX_NUM_THREADS; j++) {
                    HANDLE_ERROR(cudaMemcpy(StateT[i].inf_queue[j], StateT_Builder[i].inf_queue[j], StateT[i].n_queue[j] * sizeof(Infection), cudaMemcpyDeviceToHost));
                    HANDLE_ERROR(cudaFree(StateT_Builder[i].inf_queue[j]));
                }
                memcpy(StateT[i].n_queue, StateT_Builder[i].n_queue, MAX_NUM_THREADS * sizeof(int));
                HANDLE_ERROR(cudaFree(StateT_Builder[i].cell_inf));
                for (int j = 0; j < P.NumAdunits; j++) {
                    HANDLE_ERROR(cudaMemcpy(StateT[i].dct_queue[j], StateT_Builder[i].dct_queue[j], StateT[i].ndct_queue[j] * sizeof(ContactEvent), cudaMemcpyDeviceToHost));
                    HANDLE_ERROR(cudaFree(StateT_Builder[i].dct_queue[j]));
                }
            }
            free(StateT_Builder);
            // P:
            HANDLE_ERROR(cudaMemcpy(&P, P_GPU, sizeof(struct Param), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(P_GPU));
            // Data:
            HANDLE_ERROR(cudaMemcpy(h_data, data, sizeof(struct Data), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(data));
            s5 = h_data->s5;
            if (h_data->need_exit) {
                exit(h_data->exit_num);
            }
            free(h_data);
            /* ---                           --- */

            //// Now allocate spatial infections using Force Of Infection (s5) calculated above
            if (s5 > 0) //// if spatial infectiousness positive
            {

                // decide how many potential cell to cell infections this cell could cause
                n = (int) ignpoi_mt(s5 * sbeta * ((double) c->tot_prob),
                                    tn); //// number people this cell's population might infect elsewhere. poisson random number based on spatial infectiousness s5, sbeta (seasonality) and this cell's "probability" (guessing this is a function of its population and geographical size).
                // i2 = number of infectious people in cell c
                int i2 = c->I;

                if (n >
                    0) //// this block normalises cumulative infectiousness cell_inf by person. s5 is the total cumulative spatial infectiousness. Reason is so that infector can be chosen using ranf_mt, which returns random number between 0 and 1.
                {
                    //// normalise by cumulative spatial infectiousness.
                    for (int j = 0; j < i2 - 1; j++) StateT[tn].cell_inf[j] /= ((float) s5);
                    //// does same as the above loop just a slightly faster calculation. i.e. StateT[tn].cell_inf[i2 - 1] / s5 would equal 1 or -1 anyway.
                    StateT[tn].cell_inf[i2 - 1] = (StateT[tn].cell_inf[i2 - 1] < 0) ? -1.0f : 1.0f;
                }

                //// loop over infections to dole out. roughly speaking, this determines which infectious person in cell c infects which person elsewhere.
                for (int k = 0; k < n; k++) {
                    //// decide on infector ci/si from cell c.
                    int j; // j = index of infector
                    // if only one infectious person in cell
                    if (i2 == 1) {
                        j = 0; // infector index is first in cell (person 0)
                    }
                        // if more than one infectious person in cell pick an infectious person (given by index j)
                        //// roughly speaking, this determines which infectious person in cell c infects which person elsewhere
                    else {
                        int m;
                        s = ranf_mt(tn);    ///// choose random number between 0 and 1
                        j = m = i2 /
                                2;        ///// assign j and m to be halfway between zero and number of infected people i2 = c->I.
                        f = 1;
                        do {
                            if (m > 1)
                                m /= 2; //// amount m to change j by reduced by half. Looks like a binary search. Basically saying, keep amending potential infector j until either j less than zero or more than number of infected people until you find j s.t. spatial infectiousness "matches" s.
                            if ((j > 0) && (fabs(StateT[tn].cell_inf[j - 1]) >= s)) {
                                j -= m;
                                if (j == 0) f = 0;
                            } else if ((j < i2 - 1) && (fabs(StateT[tn].cell_inf[j]) < s)) {
                                j += m;
                                if (j == i2 - 1) f = 0;
                            } else f = 0;
                        } while (f);
                    }
                    f = (StateT[tn].cell_inf[j] <
                         0); //// flag for whether infector j had their place(s) closed. <0 (true) = place closed / >=0 (false) = place not closed. Set in if (sbeta > 0) part of loop over infectious people.
                    // ci is the index of the jth infectious person in the cell
                    ci = c->infected[j];
                    // si is the jth selected person in the cell
                    Person *si = Hosts + ci;

                    //calculate flag (fct) for digital contact tracing here at the beginning for each individual infector
                    int fct = ((P.DoDigitalContactTracing) &&
                               (t >= AdUnits[Mcells[si->mcell].adunit].DigitalContactTracingTimeStart)
                               && (t < AdUnits[Mcells[si->mcell].adunit].DigitalContactTracingTimeStart +
                                       P.DigitalContactTracingPolicyDuration) && (Hosts[ci].digitalContactTracingUser ==
                                                                                  1)); // && (ts <= (Hosts[ci].detected_time + P.usCaseIsolationDelay)));


                    //// decide on infectee

                    // do the following while f2=0
                    do {
                        //// chooses which cell person will infect
                        // pick random s between 0 and 1
                        s = ranf_mt(tn);
                        // generate l using InvCDF of selected cell and random integer between 0 and 1024
                        int l = c->InvCDF[(int) floor(s * 1024)];
                        // loop over c->cum_trans array until find a value >= random number s
                        while (c->cum_trans[l] < s) l++;
                        // selecte the cell corresponding to l
                        Cell *ct = CellLookup[l];

                        ///// pick random person m within susceptibles of cell ct (S0 initial number susceptibles within cell).
                        int m = (int) (ranf_mt(tn) * ((double) ct->S0));
                        int i3 = ct->susceptible[m];

                        s2 = dist2(Hosts + i3, Hosts +
                                               ci); /// calculate distance squared between this susceptible person and person ci/si identified earlier
                        s = P.KernelLookup.num(s2) / c->max_trans[l]; //// acceptance probability

                        // initialise f2=0 (f2=1 is the while condition for this loop)
                        f2 = 0;
                        // if random number greater than acceptance probablility or infectee is dead
                        if ((ranf_mt(tn) >= s) || (abs(Hosts[i3].inf) ==
                                                   InfStat_Dead)) //// if rejected, or infectee i3/m already dead, ensure do-while evaluated again (i.e. choose a new infectee).
                        {
                            // set f2=1 so loop continues
                            f2 = 1;
                        } else {
                            //// if potential infectee not travelling, and either is not part of cell c or doesn't share a household with infector.
                            if ((!Hosts[i3].Travelling) && ((c != ct) || (Hosts[i3].hh != si->hh))) {
                                // pick microcell of infector (mi)
                                Microcell *mi = Mcells + si->mcell;
                                // pick microcell of infectee (mt)
                                Microcell *mt = Mcells + Hosts[i3].mcell;
                                s = CalcSpatialSusc(i3, ts, ci, tn);
                                // Care home residents may have fewer contacts
                                if ((Hosts[i3].care_home_resident) || (Hosts[ci].care_home_resident))
                                    s *= P.CareHomeResidentSpatialScaling;
                                //so this person is a contact - but might not be infected. if we are doing digital contact tracing, we want to add the person to the contacts list, if both are users
                                if (fct) {
                                    //if infectee is also a user, add them as a contact
                                    if (Hosts[i3].digitalContactTracingUser && (ci != i3)) {
                                        if ((Hosts[ci].ncontacts < P.MaxDigitalContactsToTrace) &&
                                            (ranf_mt(tn) < s * P.ProportionDigitalContactsIsolate)) {
                                            Hosts[ci].ncontacts++; //add to number of contacts made
                                            int ad = Mcells[Hosts[i3].mcell].adunit;
                                            if ((StateT[tn].ndct_queue[ad] < AdUnits[ad].n)) {
                                                //find adunit for contact and add both contact and infectious host to lists - storing both so I can set times later.
                                                StateT[tn].dct_queue[ad][StateT[tn].ndct_queue[ad]++] = {i3, ci, ts};
                                            } else {
                                                fprintf(stderr_shared,
                                                        "No more space in queue! Thread: %i, AdUnit: %i\n", tn, ad);
                                            }
                                        }
                                    }
                                    //scale down susceptibility so we don't over accept
                                    s /= P.ScalingFactorSpatialDigitalContacts;
                                }
                                if (m < ct->S)  // only bother trying to infect susceptible people
                                {
                                    s *= CalcPersonSusc(i3, ts, ci, tn);
                                    if (bm) {
                                        if ((dist2_raw(Households[si->hh].loc.x, Households[si->hh].loc.y,
                                                       Households[Hosts[i3].hh].loc.x, Households[Hosts[i3].hh].loc.y) >
                                             P.MoveRestrRadius2))
                                            s *= P.MoveRestrEffect;
                                    } else if ((mt->moverest != mi->moverest) &&
                                               ((mt->moverest == 2) || (mi->moverest == 2)))
                                        s *= P.MoveRestrEffect;
                                    if ((!f) && (HOST_ABSENT(
                                            i3))) //// if infector did not have place closed, loop over place types of infectee i3 to see if their places had closed. If they had, amend their susceptibility.
                                    {
                                        for (m = f2 = 0; (m < P.PlaceTypeNum) && (!f2); m++)
                                            if (Hosts[i3].PlaceLinks[m] >= 0) {
                                                f2 = PLACE_CLOSED(m, Hosts[i3].PlaceLinks[m]);
                                            }
                                        if (f2) { s *= P.PlaceCloseSpatialRelContact; }/* NumPCD++;} */
                                        f2 = 0;
                                    }
                                    if ((s == 1) || (ranf_mt(tn) < s)) //// accept/reject
                                    {
                                        cq = ((int) (ct - Cells)) % P.NumThreads;
                                        if ((Hosts[i3].inf == InfStat_Susceptible) &&
                                            (StateT[tn].n_queue[cq] < P.InfQueuePeakLength)) //Hosts[i3].infector==-1
                                        {
                                            if ((P.FalsePositiveRate > 0) && (ranf_mt(tn) < P.FalsePositiveRate))
                                                StateT[tn].inf_queue[cq][StateT[tn].n_queue[cq]++] = {-1, i3, -1};
                                            else {
                                                short int infect_type = 2 + 2 * NUM_PLACE_TYPES + INFECT_TYPE_MASK *
                                                                                                  (1 + si->infect_type /
                                                                                                       INFECT_TYPE_MASK);
                                                StateT[tn].inf_queue[cq][StateT[tn].n_queue[cq]++] = {ci, i3,
                                                                                                      infect_type};
                                            }
                                        }
                                    }
                                }// m < susceptible people in target cell
                            }// //// if potential infectee not travelling, and either is not part of cell c or doesn't share a household with infector
                        }// infectee isn't dead
                    } while (f2);
                }// loop over infections doled out by cell
            }// s5 > 0
        }


#pragma omp parallel for schedule(static, 1) default(none) \
        shared(t, run, P, StateT, Hosts, ts)
    for (int j = 0; j < P.NumThreads; j++) {
        for (int k = 0; k < P.NumThreads; k++) {
            for (int i = 0; i < StateT[k].n_queue[j]; i++) {
                int infector = StateT[k].inf_queue[j][i].infector;
                int infectee = StateT[k].inf_queue[j][i].infectee;
                short int infect_type = StateT[k].inf_queue[j][i].infect_type;
                Hosts[infectee].infector = infector;
                Hosts[infectee].infect_type = infect_type;
                if (infect_type == -1) //// i.e. if host doesn't have an infector
                    DoFalseCase(infectee, t, ts, j);
                else
                    DoInfect(infectee, t, j, run);
            }
            StateT[k].n_queue[j] = 0;
        }
    }
}


// Error Handling.
void HANDLE_ERROR(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}
