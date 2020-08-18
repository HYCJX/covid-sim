#include "Sweep_gpu_Helper.cuh"

#include <cstdio>
#include <cmath>

#include "./Models/Cell.h"
#include "Constants.h"
#include "InfStat.h"
#include "Model.h"

/* ----- Dist ----- */

__device__ double periodic_xy_GPU(double x, double y, Param *P_GPU) {
    if (P_GPU->DoPeriodicBoundaries) {
        if (x > P_GPU->in_degrees_.width * 0.5) x = P_GPU->in_degrees_.width - x;
        if (y > P_GPU->in_degrees_.height * 0.5) y = P_GPU->in_degrees_.height - y;
    }
    return x * x + y * y;
}

__device__ double dist2UTM_GPU(double x1, double y1, double x2, double y2, Param *P_GPU) {
    double x, y, cy1, cy2, yt, xi, yi;

    x = fabs(x1 - x2) / 2;
    y = fabs(y1 - y2) / 2;
    xi = floor(x);
    yi = floor(y);
    x -= xi;
    y -= yi;
    x = (1 - x) * sinx_GPU[(int) xi] + x * sinx_GPU[((int) xi) + 1];
    y = (1 - y) * sinx_GPU[(int) yi] + y * sinx_GPU[((int) yi) + 1];
    yt = fabs(y1 + P_GPU->SpatialBoundingBox.bottom_left_GPU().y);
    yi = floor(yt);
    cy1 = yt - yi;
    cy1 = (1 - cy1) * cosx_GPU[((int) yi)] + cy1 * cosx_GPU[((int) yi) + 1];
    yt = fabs(y2 + P_GPU->SpatialBoundingBox.bottom_left_GPU().y);
    yi = floor(yt);
    cy2 = yt - yi;
    cy2 = (1 - cy2) * cosx_GPU[((int) yi)] + cy2 * cosx_GPU[((int) yi) + 1];
    x = fabs(1000 * (y * y + x * x * cy1 * cy2));
    xi = floor(x);
    x -= xi;
    y = (1 - x) * asin2sqx_GPU[((int) xi)] + x * asin2sqx_GPU[((int) xi) + 1];
    return 4 * EARTHRADIUS * EARTHRADIUS * y;
}

__device__ double dist2_raw_GPU(double ax, double ay, double bx, double by, Param *P_GPU) {
    double x, y;

    if (P_GPU->DoUTM_coords)
        return dist2UTM_GPU(ax, ay, bx, by, P_GPU);
    else {
        x = fabs(ax - bx);
        y = fabs(ay - by);
        return periodic_xy_GPU(x, y, P_GPU);
    }
}

/* ----- CalcinfSusc ----- */

// Infectiousness functions (House, Place, Spatial, Person). Idea is that in addition to a person's personal infectiousness, they have separate "infectiousnesses" for their house, place and on other cells (spatial).
// These functions consider one person only. A person has an infectiousness that is independent of other people. Slightly different therefore than susceptibility functions.
__device__ double
CalcHouseInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                 Household *Households_GPU, Param *P_GPU) {
    return ((HOST_ISOLATED_GPU(j) && (Hosts_GPU[j].digitalContactTraced != 1)) ? P_GPU->CaseIsolationHouseEffectiveness
                                                                               : 1.0)
           * ((Hosts_GPU[j].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationHouseEffectiveness : 1.0)
           * ((HOST_QUARANTINED_GPU(j) && (Hosts_GPU[j].digitalContactTraced != 1) && (!(HOST_ISOLATED_GPU(j))))
              ? P_GPU->HQuarantineHouseEffect : 1.0)
           * P_GPU->HouseholdDenomLookup[Households_GPU[Hosts_GPU[j].hh].nhr - 1]
           * ((Hosts_GPU[j].care_home_resident) ? P_GPU->CareHomeResidentHouseholdScaling : 1.0)
           * (HOST_TREATED_GPU(j) ? P_GPU->TreatInfDrop : 1.0)
           * (HOST_VACCED_GPU(j) ? P_GPU->VaccInfDrop : 1.0)
           * ((P_GPU->NoInfectiousnessSDinHH) ? ((Hosts_GPU[j].infectiousness < 0) ? P_GPU->SymptInfectiousness
                                                                                   : P_GPU->AsymptInfectiousness)
                                              : fabs(
                    Hosts_GPU[j].infectiousness))  // removed call to CalcPersonInf to allow infectiousness to be const in hh
           * P_GPU->infectiousness[ts - Hosts_GPU[j].latent_time - 1];
}

__device__ double
CalcPlaceInf_GPU(int j, int k, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                 Param *P_GPU) {
    return ((HOST_ISOLATED_GPU(j) && (Hosts_GPU[j].digitalContactTraced != 1)) ? P_GPU->CaseIsolationEffectiveness
                                                                               : 1.0)
           * ((Hosts_GPU[j].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationEffectiveness : 1.0)
           * ((HOST_QUARANTINED_GPU(j) && (!Hosts_GPU[j].care_home_resident) &&
               (Hosts_GPU[j].digitalContactTraced != 1) &&
               (!(HOST_ISOLATED_GPU(j)))) ? P_GPU->HQuarantinePlaceEffect[k] : 1.0)
           * (((Hosts_GPU[j].inf == InfStat_Case) && (!Hosts_GPU[j].care_home_resident))
              ? P_GPU->SymptPlaceTypeContactRate[k] : 1.0)
           * P_GPU->PlaceTypeTrans[k] / P_GPU->PlaceTypeGroupSizeParam1[k] * CalcPersonInf_GPU(j, ts, Hosts_GPU, P_GPU);
}

__device__ double
CalcSpatialInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU,
                   Param *P_GPU) {
    return ((HOST_ISOLATED_GPU(j) && (Hosts_GPU[j].digitalContactTraced != 1)) ? P_GPU->CaseIsolationEffectiveness
                                                                               : 1.0)
           * ((Hosts_GPU[j].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationEffectiveness : 1.0)
           * ((HOST_QUARANTINED_GPU(j) && (!Hosts_GPU[j].care_home_resident) &&
               (Hosts_GPU[j].digitalContactTraced != 1) &&
               (!(HOST_ISOLATED_GPU(j)))) ? P_GPU->HQuarantineSpatialEffect : 1.0)
           * ((Hosts_GPU[j].inf == InfStat_Case) ? P_GPU->SymptSpatialContactRate : 1.0)
           * P_GPU->RelativeSpatialContact[HOST_AGE_GROUP_GPU(j)]
           * CalcPersonInf_GPU(j, ts, Hosts_GPU, P_GPU);        /*	*Hosts[j].spatial_norm */
}

__device__ double CalcPersonInf_GPU(int j, unsigned short int ts, Person *Hosts_GPU, Param *P_GPU) {
    return (HOST_TREATED_GPU(j) ? P_GPU->TreatInfDrop : 1.0)
           * (HOST_VACCED_GPU(j) ? P_GPU->VaccInfDrop : 1.0)
           * fabs(Hosts_GPU[j].infectiousness)
           * P_GPU->infectiousness[ts - Hosts_GPU[j].latent_time - 1];
}

// Susceptibility functions (House, Place, Spatial, Person). Similarly, idea is that in addition to a person's personal susceptibility, they have separate "susceptibilities" for their house, place and on other cells (spatial)
// These functions consider two people. A person has a susceptibility TO ANOTHER PERSON/infector. Slightly different therefore than infectiousness functions.
__device__ double
CalcHouseSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU, Microcell *Mcells_GPU,
                  Param *P_GPU) {
    return CalcPersonSusc_GPU(ai, ts, infector, tn, Hosts_GPU, P_GPU)
           * ((Mcells_GPU[Hosts_GPU[ai].mcell].socdist == 2) ? ((Hosts_GPU[ai].esocdist_comply)
                                                                ? P_GPU->EnhancedSocDistHouseholdEffectCurrent
                                                                : P_GPU->SocDistHouseholdEffectCurrent) : 1.0)
           * ((Hosts_GPU[ai].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationHouseEffectiveness : 1.0)
           * ((Hosts_GPU[ai].care_home_resident) ? P_GPU->CareHomeResidentHouseholdScaling : 1.0);
}

__device__ double CalcPlaceSusc_GPU(int ai, int k, unsigned short int ts, int infector, int tn, Person *Hosts_GPU,
                                    PersonQuarantine *HostsQuarantine_GPU, Microcell *Mcells_GPU, Param *P_GPU) {
    return ((HOST_QUARANTINED_GPU(ai) && (!Hosts_GPU[ai].care_home_resident) &&
             (Hosts_GPU[ai].digitalContactTraced != 1))
            ? P_GPU->HQuarantinePlaceEffect[k] : 1.0)
           * ((Mcells_GPU[Hosts_GPU[ai].mcell].socdist == 2) ? ((Hosts_GPU[ai].esocdist_comply)
                                                                ? P_GPU->EnhancedSocDistPlaceEffectCurrent[k]
                                                                : P_GPU->SocDistPlaceEffectCurrent[k]) : 1.0)
           * ((Hosts_GPU[ai].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationEffectiveness : 1.0);
}

__device__ double CalcSpatialSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU,
                                      PersonQuarantine *HostsQuarantine_GPU, Microcell *Mcells_GPU, Param *P_GPU) {
    return ((HOST_QUARANTINED_GPU(ai) && (!Hosts_GPU[ai].care_home_resident) &&
             (Hosts_GPU[ai].digitalContactTraced != 1))
            ? P_GPU->HQuarantineSpatialEffect : 1.0)
           * ((Mcells_GPU[Hosts_GPU[ai].mcell].socdist == 2) ? ((Hosts_GPU[ai].esocdist_comply)
                                                                ? P_GPU->EnhancedSocDistSpatialEffectCurrent
                                                                : P_GPU->SocDistSpatialEffectCurrent) : 1.0)
           * ((Hosts_GPU[ai].digitalContactTraced == 1) ? P_GPU->DCTCaseIsolationEffectiveness : 1.0);
}

__device__ double
CalcPersonSusc_GPU(int ai, unsigned short int ts, int infector, int tn, Person *Hosts_GPU, Param *P_GPU) {
    return P_GPU->WAIFW_Matrix[HOST_AGE_GROUP_GPU(ai)][HOST_AGE_GROUP_GPU(infector)]
           * P_GPU->AgeSusceptibility[HOST_AGE_GROUP_GPU(ai)] * Hosts_GPU[ai].susc
           * (HOST_TREATED_GPU(ai) ? P_GPU->TreatSuscDrop : 1.0)
           * (HOST_VACCED_GPU(ai) ? (HOST_VACCED_SWITCH_GPU(ai) ? P_GPU->VaccSuscDrop2 : P_GPU->VaccSuscDrop) : 1.0);
}

/* ----- Computation Kernel ----- */

__global__ void
kernel(double t, int tn, Cell *c, Person *Hosts_GPU, PersonQuarantine *HostsQuarantine_GPU, Household *Households_GPU,
       Microcell *Mcells_GPU, Place **Places_GPU, AdminUnit *AdUnits_GPU, int **SamplingQueue_GPU, PopVar *StateT_GPU,
       Param *P_GPU, Data *data) {

    /* Variables */
    int n ; // Number of people you could potentially infect in your place group, then number of potential spatial infections doled out by cell on other cells.
    int f, f2, cq /* Cell queue */, bm = data->bm /* Movement restrictions in place */, ci /* Person index */;
    double s; // Household Force Of Infection (FOI) on fellow household member, then place susceptibility, then random number for spatial infections allocation.
    double s2; // Spatial infectiousness, then distance in spatial infections allocation.
    double s3, s3_scaled; // Household, then place infectiousness.
    double s4, s4_scaled; // Place infectiousness (copy of s3 as some code commented out.
    double s5 = data->s5; // Total spatial infectiousness summed over all infectious people in cell.
    double s6;
    double seasonality = data->seasonality, sbeta = data->sbeta, hbeta = data->hbeta;
    double fp = data->fp; // False positive.
    unsigned short int ts = data->ts;
    FILE *stderr_shared = stderr;

    /* Thread id */
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    //// get person index ci of j'th infected person in cell
    ci = c->infected[j];
    //// get person si from Hosts (array of people) corresponding to ci, using pointer arithmetic.
    Person *si = Hosts_GPU + ci;

    /* Evaluate flag for digital contact tracing (fct) here at the beginning for each individual.
     * fct = 1 if:
     * P.DoDigitalContactTracing = 1 (ie. digital contact tracing functionlity is switched on)
     * AND Day number (t) is greater than the start day for contact tracing in this administrative unit (ie. contact tracing has started)
     * AND Day number (t) is less than the end day for contact tracing in this administrative unit (ie. contact tracing has not ended)
     * AND the selected host is a digital contact tracing user.
     * otherwise fct = 0.
     */
    int fct = ((P_GPU->DoDigitalContactTracing) &&
               (t >= AdUnits_GPU[Mcells_GPU[si->mcell].adunit].DigitalContactTracingTimeStart)
               && (t < AdUnits_GPU[Mcells_GPU[si->mcell].adunit].DigitalContactTracingTimeStart +
                       P_GPU->DigitalContactTracingPolicyDuration) && (Hosts_GPU[ci].digitalContactTracingUser ==
                                                                       1)); // && (ts <= (Hosts[ci].detected_time + P.usCaseIsolationDelay)));

    // BEGIN HOUSEHOLD INFECTIONS

    //// Household Force Of Infection (FOI) component

    // hbeta =  seasonality * fp * P.HouseholdTrans or 0 depending on whether households functionality is on or off - see start of function

    // if household beta (hbeta) > 0
    if (hbeta > 0) {
        // For selected host si's household (si->hh),
        // if the number of hosts (nh) in that Household is greater than 1
        // AND the selected host is not travelling
        if ((Households_GPU[si->hh].nh > 1) && (!si->Travelling)) {
            int l = Households_GPU[si->hh].FirstPerson;
            int m = l + Households_GPU[si->hh].nh;
            // calculate infectiousness of selected household (s3)
            // using the CalcHouseInf function on the selected cell and timestamp at start of current day
            // then scaling by hbeta
            s3 = hbeta * CalcHouseInf_GPU(ci, ts, Hosts_GPU, HostsQuarantine_GPU, Households_GPU, P_GPU);

            // Test if any of the individuals in the selected persons household are absent from places
            // f=0 means noone absent, f=1 means at least one absent
            f = 0; // initialise f to be 0
            for (int i3 = l; (i3 < m) && (!f); i3++) { //// loop over people in household
                for (int i2 = 0; (i2 < P_GPU->PlaceTypeNum) && (!f); i2++) { //// loop over place types
                    if (Hosts_GPU[i3].PlaceLinks[i2] >=
                        0) { //// if person in household has any sort of link to place type
                        // if person is absent set f=1
                        f = ((PLACE_CLOSED_GPU(i2, Hosts_GPU[i3].PlaceLinks[i2])) && (HOST_ABSENT_GPU(i3)));
                    }
                }
            }

            // if individuals in the household are absent from places (ie. f==1 from test immediately above), scale up the infectiousness (s3) of the household
            if (f) { s3 *= P_GPU->PlaceCloseHouseholdRelContact; }/* NumPCD++;}*/ //// if people in your household are absent from places, person si/ci is more infectious to them, as they spend more time at home.

            // Loop from l (the index of the first person in the household) to m-1 (the index of the last person in the household)
            // ie. loop over everyone in the household
            for (int i3 = l; i3 < m; i3++) //// loop over all people in household (note goes from l to m - 1)
            {
                if ((Hosts_GPU[i3].inf == InfStat_Susceptible) &&
                    (!Hosts_GPU[i3].Travelling)) //// if people in household uninfected/susceptible and not travelling
                {
                    s = s3 * CalcHouseSusc_GPU(i3, ts, ci,
                                               tn, Hosts_GPU, Mcells_GPU,
                                               P_GPU);        //// FOI ( = infectiousness x susceptibility) from person ci/si on fellow household member i3

                    // Force of Infection (s) > random value between 0 and 1
                    if (ranf_mt(tn) < s) {
                        // identify which cell queue (index cq) to add infection to
                        cq = Hosts_GPU[i3].pcell % P_GPU->NumThreads;
                        if ((StateT_GPU[tn].n_queue[cq] < P_GPU->InfQueuePeakLength)) //(Hosts[i3].infector==-1)&&
                        {
                            if ((P_GPU->FalsePositiveRate > 0) && (ranf_mt(tn) < P_GPU->FalsePositiveRate))
                                StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {-1, i3, -1};
                            else {

                                // ** infect household member i3 **
                                Hosts_GPU[i3].infector = ci; //// assign person ci as infector of person i3
                                //infect_type: first 4 bits store type of infection
                                //				1= household
                                //				2..NUM_PLACE_TYPES+1 = within-class/work-group place based transmission
                                //				NUM_PLACE_TYPES+2..2*NUM_PLACE_TYPES+1 = between-class/work-group place based transmission
                                //				2*NUM_PLACE_TYPES+2 = "spatial" transmission (spatially local random mixing)
                                // bits >4 store the generation of infection

                                short int infect_type = 1 + INFECT_TYPE_MASK * (1 + si->infect_type / INFECT_TYPE_MASK);
                                StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {ci, i3, infect_type};
                            }
                        }
                    }// if FOI > s
                } // if person in household uninfected/susceptible and not travelling
            }// loop over people in household
        } // if more than one person in household
    }// if hbeta > 0

    // END HOUSHOLD INFECTIONS

    // BEGIN PLACE INFECTIONS

    // Still with infected person (si) = Hosts[ci]
    // if places functionality is enabled
    if (P_GPU->DoPlaces) {
        // if host with index ci isn't absent
        if (!HOST_ABSENT_GPU(ci)) {
            // select microcell (mi) corresponding to selected host (si)
            Microcell *mi = Mcells_GPU + si->mcell;
            for (int k = 0; k < P_GPU->PlaceTypeNum; k++) //// loop over all place types
            {
                // select link (l) between selected host (si) and place from si's placelinks to place type k
                int l = si->PlaceLinks[k];
                if (l >=
                    0)  //// l>=0 means if place type k is relevant to person si. (Now allowing for partial attendance).
                {
                    // infectiousness of place (s3)
                    // = false positive rate * seasonality * place infectiousness
                    s3 = fp * seasonality * CalcPlaceInf_GPU(ci, k, ts, Hosts_GPU, HostsQuarantine_GPU, P_GPU);
                    // select microcell of the place linked to host si with link l
                    Microcell *mp = Mcells_GPU + Places_GPU[k][l].mcell;
                    // if blanket movement restrictions are in place on current day
                    if (bm) {
                        // if distance between si's household and linked place
                        // is greater than movement restriction radius
                        if ((dist2_raw_GPU(Households_GPU[si->hh].loc.x, Households_GPU[si->hh].loc.y,
                                           Places_GPU[k][l].loc.x, Places_GPU[k][l].loc.y, P_GPU) >
                             P_GPU->MoveRestrRadius2)) {
                            // multiply infectiousness of place by movement restriction effect
                            s3 *= P_GPU->MoveRestrEffect;
                        }
                    }
                        // else if movement restrictions in effect in either household microcell or place microcell
                    else if ((mi->moverest != mp->moverest) && ((mi->moverest == 2) || (mp->moverest == 2))) {
                        // multiply infectiousness of place by movement restriction effect
                        s3 *= P_GPU->MoveRestrEffect;
                    }

                    // BEGIN NON-HOTEL INFECTIONS

                    // if linked place isn't a hotel and selected host isn't travelling
                    if ((k != P_GPU->HotelPlaceType) && (!si->Travelling)) {
                        // i2 is index of group (of place type k) that selected host is linked to
                        int i2 = (si->PlaceGroupLinks[k]);

                        // calculate infectiousness (s4_scaled)
                        // which varies if contact tracing is in place
                        // if contact tracing isn't in place s4_scaled is a copy of s3
                        // if contact tracing is in place, s4_scaled is s3  * P.ScalingFactorPlaceDigitalContacts
                        // in either case s4_scaled is capped at 1

                        // if contact tracing
                        if (fct) {
                            // copy s3
                            s4 = s3;
                            // multiply s4 by P.ScalingFactorPlaceDigitalContacts
                            s4_scaled = s4 * P_GPU->ScalingFactorPlaceDigitalContacts;
                            // cap s4 at 1
                            if (s4 > 1) s4 = 1;
                            // cap at 1
                            if (s4_scaled > 1) s4_scaled = 1;
                        } else {
                            // copy s3 to s4
                            s4 = s3;
                            // cap s4 at 1
                            if (s4 > 1) s4 = 1;
                            s4_scaled = s4;
                        }

                        // if infectiousness is < 0, we have an error - end the program
                        if (s4_scaled < 0) {
                            // fprintf(stderr_shared, "@@@ %lg\n", s4_scaled);
                            need_exit = true;
                            exit_num = 1;
                            return;
                        }
                            // else if infectiousness == 1 (should never be more than 1 due to capping above)
                        else if (s4_scaled >=
                                 1)    //// if place infectiousness above threshold, consider everyone in group a potential infectee...
                        {
                            // set n to be number of people in group in place k,l
                            n = Places_GPU[k][l].group_size[i2];
                        } else                //// ... otherwise randomly sample (from binomial distribution) number of potential infectees in this place.
                        {
                            n = (int) ignbin_mt((int32_t) Places_GPU[k][l].group_size[i2], s4_scaled, tn);
                        }

                        // if potential infectees > 0
                        if (n > 0) {
                            // pick n members of place k,l and add them to sampling queue for thread tn
                            SampleWithoutReplacement(tn, n,
                                                     Places_GPU[k][l].group_size[i2]); //// changes thread-specific SamplingQueue.
                        }

                        // loop over sampling queue of potential infectees
                        for (int m = 0; m < n; m++) {
                            // pick potential infectee index i3
                            int i3 = Places_GPU[k][l].members[Places_GPU[k][l].group_start[i2] +
                                                              SamplingQueue_GPU[tn][m]];
                            // calculate place susceptbility based on infectee (i3), place type (k), timestep (ts)
                            // cell (ci) and thread number (tn)
                            s = CalcPlaceSusc_GPU(i3, k, ts, ci, tn, Hosts_GPU, HostsQuarantine_GPU, Mcells_GPU, P_GPU);

                            // allow care home residents to mix more intensely in "groups" (i.e. individual homes) than staff do - to allow for PPE/environmental contamination.
                            if ((k == P_GPU->CareHomePlaceType) &&
                                ((!Hosts_GPU[ci].care_home_resident) || (!Hosts_GPU[i3].care_home_resident)))
                                s *= P_GPU->CareHomeWorkerGroupScaling;
                            //these are all place group contacts to be tracked for digital contact tracing - add to StateT queue for contact tracing
                            //if infectee is also a user, add them as a contact

                            if ((fct) && (Hosts_GPU[i3].digitalContactTracingUser) && (ci != i3) &&
                                (!HOST_ABSENT_GPU(i3))) {
                                // scale place susceptibility by proportion who self isolate and store as s6
                                s6 = P_GPU->ProportionDigitalContactsIsolate * s;
                                // if random number < s6
                                // AND number of contacts of ci(!) is less than maximum digital contact to trace
                                if ((Hosts_GPU[ci].ncontacts < P_GPU->MaxDigitalContactsToTrace) &&
                                    (ranf_mt(tn) < s6)) {
                                    Hosts_GPU[ci].ncontacts++; //add to number of contacts made
                                    int ad = Mcells_GPU[Hosts_GPU[i3].mcell].adunit;
                                    if ((StateT_GPU[tn].ndct_queue[ad] < AdUnits_GPU[ad].n)) {
                                        //find adunit for contact and add both contact and infectious host to lists - storing both so I can set times later.
                                        StateT_GPU[tn].dct_queue[ad][StateT_GPU[tn].ndct_queue[ad]++] = {i3, ci, ts};
                                    } else {
//                                        fprintf(stderr_shared, "No more space in queue! Thread: %i, AdUnit: %i\n", tn,
//                                                ad);
                                    }
                                }
                            }

                            if ((Hosts_GPU[i3].inf == InfStat_Susceptible) &&
                                (!HOST_ABSENT_GPU(i3))) //// if person i3 uninfected and not absent.
                            {
                                Microcell *mt = Mcells_GPU + Hosts_GPU[i3].mcell;
                                //downscale s if it has been scaled up do to digital contact tracing
                                s *= CalcPersonSusc_GPU(i3, ts, ci, tn, Hosts_GPU, P_GPU) * s4 / s4_scaled;

                                // if blanket movement restrictions are in place
                                if (bm) {
                                    // if potential infectee i3's household is further from selected place
                                    if ((dist2_raw_GPU(Households_GPU[Hosts_GPU[i3].hh].loc.x,
                                                       Households_GPU[Hosts_GPU[i3].hh].loc.y,
                                                       Places_GPU[k][l].loc.x, Places_GPU[k][l].loc.y, P_GPU) >
                                         P_GPU->MoveRestrRadius2)) {
                                        // multiply susceptibility by movement restriction effect
                                        s *= P_GPU->MoveRestrEffect;
                                    }
                                }
                                    // else if movement restrictions are in place in either cell
                                else if ((mt->moverest != mp->moverest) &&
                                         ((mt->moverest == 2) || (mp->moverest == 2))) {
                                    // multiply susceptibility by movement restriction effect
                                    s *= P_GPU->MoveRestrEffect;
                                }

                                // if either susceptiblity is 100% or sample probability s
                                if ((s == 1) || (ranf_mt(tn) < s)) {
                                    // select cell containing potential infectee
                                    cq = Hosts_GPU[i3].pcell % P_GPU->NumThreads;
                                    // if infection queue for selected call < maximum length
                                    if ((StateT_GPU[tn].n_queue[cq] <
                                         P_GPU->InfQueuePeakLength)) //(Hosts[i3].infector==-1)&&
                                    {
                                        // false positive
                                        if ((P_GPU->FalsePositiveRate > 0) && (ranf_mt(tn) < P_GPU->FalsePositiveRate))
                                            StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {-1, i3, -1};
                                        else {
                                            // infect i3 - add if to infection queue for selected cell
                                            short int infect_type =
                                                    2 + k + INFECT_TYPE_MASK * (1 + si->infect_type / INFECT_TYPE_MASK);
                                            StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {ci, i3,
                                                                                                          infect_type};
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // END NON-HOTEL INFECTIONS

                    // BEGIN HOTEL INFECTIONS

                    // if selected host si is not travelling or selected link is to a hotel
                    if ((k == P_GPU->HotelPlaceType) || (!si->Travelling)) {
                        s3 *= P_GPU->PlaceTypePropBetweenGroupLinks[k] * P_GPU->PlaceTypeGroupSizeParam1[k] /
                              ((double) Places_GPU[k][l].n);
                        if (s3 > 1) s3 = 1;
                        // if contact tracing in place, multiply s3_scaled = s3*scalingfactor, otherwise s3_scaled = s3
                        s3_scaled = (fct) ? (s3 * P_GPU->ScalingFactorPlaceDigitalContacts) : s3;
                        // s3_scales shouldn't be less than 0 so generate error if it is
                        if (s3_scaled < 0) {
//                            ERR_CRITICAL_FMT("@@@ %lg\n", s3);
                        }
                            // if s3_scaled >=1, everyone in the hotel is a potential infectee
                        else if (s3_scaled >= 1)
                            n = Places_GPU[k][l].n;
                            // if s3_scaled between 0 and 1, decide number of potential infectees based on
                            // using ignbin_mt function
                        else
                            n = (int) ignbin_mt((int32_t) Places_GPU[k][l].n, s3_scaled, tn);
                        // if more than 0 potential infectees, pick n hosts from the hotel and add to sampling queue
                        if (n > 0) SampleWithoutReplacement(tn, n, Places_GPU[k][l].n);
                        // loop over the sampling queue
                        for (int m = 0; m < n; m++) {
                            // select potential infectee from sampling queue
                            int i3 = Places_GPU[k][l].members[SamplingQueue_GPU[tn][m]];
                            // calculate place susceptibility s
                            s = CalcPlaceSusc_GPU(i3, k, ts, ci, tn, Hosts_GPU, HostsQuarantine_GPU, Mcells_GPU, P_GPU);
                            // use group structure to model multiple care homes with shared staff - in which case residents of one "group" don't mix with those in another, only staff do.
                            if ((Hosts_GPU[ci].care_home_resident) && (Hosts_GPU[i3].care_home_resident) &&
                                (Hosts_GPU[ci].PlaceGroupLinks[k] != Hosts_GPU[i3].PlaceGroupLinks[k]))
                                s *= P_GPU->CareHomeResidentPlaceScaling;
                            // allow care home staff to have lowere contacts in care homes - to allow for PPE/environmental contamination.
                            if ((k == P_GPU->CareHomePlaceType) &&
                                ((!Hosts_GPU[ci].care_home_resident) || (!Hosts_GPU[i3].care_home_resident)))
                                s *= P_GPU->CareHomeWorkerGroupScaling;

                            //these are all place group contacts to be tracked for digital contact tracing - add to StateT queue for contact tracing
                            //if infectee is also a user, add them as a contact

                            // if contact tracing in place AND potential infectee i3 is a contact tracing user AND i3 isn't absent AND i3 isn't ci (suspect this should be si)
                            if ((fct) && (Hosts_GPU[i3].digitalContactTracingUser) && (ci != i3) &&
                                (!HOST_ABSENT_GPU(i3))) {
                                // s6 = place susceptibility * proportion of digital contacts who self isolate
                                s6 = P_GPU->ProportionDigitalContactsIsolate * s;
                                // if number of contacts of infectious person < maximum and random number < s6
                                if ((Hosts_GPU[ci].ncontacts < P_GPU->MaxDigitalContactsToTrace) &&
                                    (ranf_mt(tn) < s6)) {
                                    Hosts_GPU[ci].ncontacts++; //add to number of contacts made
                                    int ad = Mcells_GPU[Hosts_GPU[i3].mcell].adunit;
                                    if ((StateT_GPU[tn].ndct_queue[ad] < AdUnits_GPU[ad].n)) {
                                        //find adunit for contact and add both contact and infectious host to lists - storing both so I can set times later.
                                        StateT_GPU[tn].dct_queue[ad][StateT_GPU[tn].ndct_queue[ad]++] = {i3, ci, ts};
                                    } else {
//                                        fprintf(stderr_shared, "No more space in queue! Thread: %i, AdUnit: %i\n", tn,
//                                                ad);
                                    }
                                }
                            }

                            // if potential infectee i3 uninfected and not absent.
                            if ((Hosts_GPU[i3].inf == InfStat_Susceptible) && (!HOST_ABSENT_GPU(i3))) {
                                // mt = microcell of potential infectee
                                Microcell *mt = Mcells_GPU + Hosts_GPU[i3].mcell;

                                //if doing digital contact tracing, scale down susceptibility here
                                s *= CalcPersonSusc_GPU(i3, ts, ci, tn, Hosts_GPU, P_GPU) * s3 / s3_scaled;
                                // if blanket movement restrictions are in place
                                if (bm) {
                                    // if potential infectees household is farther away from hotel than restriction radius
                                    if ((dist2_raw_GPU(Households_GPU[Hosts_GPU[i3].hh].loc.x,
                                                       Households_GPU[Hosts_GPU[i3].hh].loc.y,
                                                       Places_GPU[k][l].loc.x, Places_GPU[k][l].loc.y, P_GPU) >
                                         P_GPU->MoveRestrRadius2)) {
                                        // multiply susceptibility by movement restriction effect
                                        s *= P_GPU->MoveRestrEffect;
                                    }
                                }
                                    // else if movement restrictions are in place in potential infectee's cell or hotel's cell
                                else if ((mt->moverest != mp->moverest) &&
                                         ((mt->moverest == 2) || (mp->moverest == 2))) {
                                    // multiply susceptibility by movement restriction effect
                                    s *= P_GPU->MoveRestrEffect;
                                }

                                // ** do infections **

                                // is susceptibility is 1 (ie infect everyone) or random number is less than susceptibility
                                if ((s == 1) || (ranf_mt(tn) < s)) {
                                    // store cell number of potential infectee i3 as cq
                                    cq = Hosts_GPU[i3].pcell % P_GPU->NumThreads;
                                    // if there is space in queue for this thread
                                    if ((StateT_GPU[tn].n_queue[cq] <
                                         P_GPU->InfQueuePeakLength))//(Hosts[i3].infector==-1)&&
                                    {
                                        // if random number < false positive rate
                                        if ((P_GPU->FalsePositiveRate > 0) && (ranf_mt(tn) < P_GPU->FalsePositiveRate))
                                            // add false positive to infection queue
                                            StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {-1, i3, -1};
                                        else {
                                            short int infect_type = 2 + k + NUM_PLACE_TYPES + INFECT_TYPE_MASK * (1 +
                                                                                                                  si->infect_type /
                                                                                                                  INFECT_TYPE_MASK);
                                            // add infection of i3 by ci to infection queue
                                            StateT_GPU[tn].inf_queue[cq][StateT_GPU[tn].n_queue[cq]++] = {ci, i3,
                                                                                                          infect_type};
                                        }
                                    }// space in queue
                                }// susceptibility test
                            }// potential infectee i3 uninfected and not absent.
                        }// loop over sampling queue
                    }// selected host si is not travelling or selected link is to a hotel

                    // ** END HOTEL INFECTIONS **

                }// if place link relevant
            }// loop over place types
        }// if host isn't absent
    }// if places functionality enabled

    // END PLACE INFECTIONS

    // BEGIN SPATIAL INFECTIONS

    //// First determine spatial FOI component (s5)

    // if seasonality beta > 0
    // do spatial infections
    //// ie sum spatial infectiousness over all infected people, the infections from which are allocated after loop over infected people.
    if (sbeta > 0) {
        if (si->Travelling) //// if host currently away from their cell, they cannot add to their cell's spatial infectiousness.
        {
            s2 = 0;
            f = 0;
        } else {
            // calculate spatial infectiousness (s2) based on host and timestep
            s2 = CalcSpatialInf_GPU(ci, ts, Hosts_GPU, HostsQuarantine_GPU, P_GPU);
            //if do digital contact tracing, scale up spatial infectiousness of infectives who are using the app and will be detected
            if (fct) {
                s2 *= P_GPU->ScalingFactorSpatialDigitalContacts;
            }
        }
        // test if selected person si is linked to a place that is closed, f=0 means no links to closed places, otherwise f=1
        f = 0; // initialise f as 0
        // If place functionality switched on
        if (P_GPU->DoPlaces) {
            // loop over place types until closed place is found
            for (int i3 = 0; (i3 < P_GPU->PlaceTypeNum) && (!f); i3++) {
                if (si->PlaceLinks[i3] >= 0) //// if person has a link to place of type i3...
                {
                    // if place is closed set f=1
                    f = PLACE_CLOSED_GPU(i3, si->PlaceLinks[i3]); //// find out if that place of type i3 is closed.
                }
            }
        }// if doing places

        if ((f) && (HOST_ABSENT_GPU(
                ci))) //// if place is closed and person is absent then adjust the spatial infectiousness (similar logic to household infectiousness: place closure affects spatial infectiousness
        {
            s2 *= P_GPU->PlaceCloseSpatialRelContact;
            /* NumPCD++; */
            s5 += s2;
            StateT_GPU[tn].cell_inf[j] = (float) -s5;
        } else {
            s5 += s2;
            StateT_GPU[tn].cell_inf[j] = (float) s5;
        }
    }

}