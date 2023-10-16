/**
 * @file    integrator_whfast.c
 * @brief   WHFAST512 integration scheme.
 * @author  Hanno Rein <hanno@hanno-rein.de>
 *          Pejvak Javaheri <pejvak.javaheri@mail.utoronto.ca>
 * @details This file implements the WHFast512 integration scheme with 
 * optimizations for AVX512.
 * 
 * @section LICENSE
 * Copyright (c) 2023 Hanno Rein, Pejvak Javaheri
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include "rebound.h"
#include "particle.h"
#include "tools.h"
#include "gravity.h"
#include "boundary.h"
#include "integrator.h"
#include "integrator_whfast.h"
#include "integrator_whfast512.h"

#ifdef AVX512

#ifdef PROF
// Profiling counters
double walltime_interaction=0;;
double walltime_kepler=0;
double walltime_jump=0;
double walltime_com=0;
#endif

// Fast inverse factorial lookup table
static const double invfactorial[35] = {1., 1., 1./2., 1./6., 1./24., 1./120., 1./720., 1./5040., 1./40320., 1./362880., 1./3628800., 1./39916800., 1./479001600., 1./6227020800., 1./87178291200., 1./1307674368000., 1./20922789888000., 1./355687428096000., 1./6402373705728000., 1./121645100408832000., 1./2432902008176640000., 1./51090942171709440000., 1./1124000727777607680000., 1./25852016738884976640000., 1./620448401733239439360000., 1./15511210043330985984000000., 1./403291461126605635584000000., 1./10888869450418352160768000000., 1./304888344611713860501504000000., 1./8841761993739701954543616000000., 1./265252859812191058636308480000000., 1./8222838654177922817725562880000000., 1./263130836933693530167218012160000000., 1./8683317618811886495518194401280000000., 1./295232799039604140847618609643520000000.};

// Vector constants
static __m512d invfactorial512[35];
static __m512d gr_prefac;
static __m512d gr_prefac2;
static __m512d half;
static __m512d one;
static __m512d two;
static __m512d five;
static __m512d sixteen;
static __m512d twenty;
static __m512d _M;
//static __m512i so2; // cross lane permutations
//static __m512i so1; 

//LEAVE FOR NOW
// Debug function to print vectors
void static inline printavx512(__m512d a) {
    double _nax[8];
    _mm512_store_pd(&_nax[0], a);
    printf("%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e    <-- avx\n", _nax[0], _nax[1], _nax[2], _nax[3], _nax[4], _nax[5], _nax[6], _nax[7]);
}
//LEAVE FOR NOW
// Stiefel function for Newton's method, returning Gs1, Gs2, and Gs3
static void inline mm_stiefel_Gs13_avx512(__m512d * Gs1, __m512d * Gs2, __m512d * Gs3, __m512d beta, __m512d X){
    __m512d X2 = _mm512_mul_pd(X,X); 
    __m512d z = _mm512_mul_pd(X2,beta); 

    // stumpff_cs. Note: assuming n = 0
    const int nmax = 19;
   *Gs3 = invfactorial512[nmax]; 
   *Gs2 = invfactorial512[nmax-1]; 

    for(int np=nmax-2;np>=3;np-=2){
        *Gs3 = _mm512_fnmadd_pd(z, *Gs3, invfactorial512[np]);
        *Gs2 = _mm512_fnmadd_pd(z, *Gs2, invfactorial512[np-1]);
    }
    *Gs3 = _mm512_mul_pd(*Gs3,X); 
    *Gs1 = _mm512_fnmadd_pd(z, *Gs3, X);
    *Gs3 = _mm512_mul_pd(*Gs3,X2); 
    *Gs2 = _mm512_mul_pd(*Gs2,X2); 
};
//LEAVE FOR NOW
// Stiefel function for Halley's method, returning Gs0, Gs1, Gs2, and Gs3
static void inline mm_stiefel_Gs03_avx512(__m512d * Gs0, __m512d * Gs1, __m512d * Gs2, __m512d * Gs3, __m512d beta, __m512d X){
    __m512d X2 = _mm512_mul_pd(X,X); 
    __m512d z = _mm512_mul_pd(X2,beta); 

    // stumpff_cs. Note: assuming n = 0
    const int nmax = 11; // Note: reduced! needs to be improved with mm_stiefel_Gs13_avx512 on last step(s)
   *Gs3 = invfactorial512[nmax]; 
   *Gs2 = invfactorial512[nmax-1]; 

    for(int np=nmax-2;np>=3;np-=2){
        *Gs3 = _mm512_fnmadd_pd(z, *Gs3, invfactorial512[np]);
        *Gs2 = _mm512_fnmadd_pd(z, *Gs2, invfactorial512[np-1]);
    }
    *Gs0 = _mm512_fnmadd_pd(z, *Gs2, one);
    *Gs3 = _mm512_mul_pd(*Gs3,X); 
    *Gs1 = _mm512_fnmadd_pd(z, *Gs3, X);
    *Gs3 = _mm512_mul_pd(*Gs3,X2); 
    *Gs2 = _mm512_mul_pd(*Gs2,X2); 
};

// Performs one full Kepler step
static void inline reb_whfast512_kepler_step(struct reb_particle_avx512* const p_jh, const double dt){
#ifdef PROF
    struct timeval time_beginning;
    gettimeofday(&time_beginning,NULL);
#endif
    struct reb_particle_avx512 * restrict const p512  = p_jh;
    __m512d _dt = _mm512_set1_pd(dt); 
        
    __m512d r2 = _mm512_mul_pd(p512->x, p512->x);
    r2 = _mm512_fmadd_pd(p512->y, p512->y, r2);
    r2 = _mm512_fmadd_pd(p512->z, p512->z, r2);
    __m512d r0 = _mm512_sqrt_pd(r2);
    __m512d r0i = _mm512_div_pd(one,r0);

    __m512d v2 = _mm512_mul_pd(p512->vx, p512->vx);
    v2 = _mm512_fmadd_pd(p512->vy, p512->vy, v2);
    v2 = _mm512_fmadd_pd(p512->vz, p512->vz, v2);
    
    __m512d beta = _mm512_mul_pd(two, _M);
    beta = _mm512_fmsub_pd(beta, r0i, v2);

    __m512d eta0 = _mm512_mul_pd(p512->x, p512->vx);
    eta0 = _mm512_fmadd_pd(p512->y, p512->vy, eta0);
    eta0 = _mm512_fmadd_pd(p512->z, p512->vz, eta0);

    __m512d zeta0 = _mm512_fnmadd_pd(beta, r0, _M);

    __m512d Gs1;
    __m512d Gs2;
    __m512d Gs3;
    __m512d eta0Gs1zeta0Gs2; 
    __m512d ri; 

#define NEWTON_STEP() \
    mm_stiefel_Gs13_avx512(&Gs1, &Gs2, &Gs3, beta, X);\
    eta0Gs1zeta0Gs2 = _mm512_mul_pd(eta0, Gs1); \
    eta0Gs1zeta0Gs2 = _mm512_fmadd_pd(zeta0,Gs2, eta0Gs1zeta0Gs2); \
    ri = _mm512_add_pd(r0, eta0Gs1zeta0Gs2); \
    ri = _mm512_div_pd(one, ri); \
    \
    X = _mm512_mul_pd(X, eta0Gs1zeta0Gs2);\
    X = _mm512_fnmadd_pd(eta0, Gs2, X);\
    X = _mm512_fnmadd_pd(zeta0, Gs3, X);\
    X = _mm512_add_pd(_dt, X);\
    X = _mm512_mul_pd(ri, X);


#define HALLEY_STEP() \
    mm_stiefel_Gs03_avx512(&Gs0, &Gs1, &Gs2, &Gs3, beta, X);\
    f = _mm512_fmsub_pd(r0,X,_dt);\
    f = _mm512_fmadd_pd(eta0, Gs2, f);\
    f = _mm512_fmadd_pd(zeta0, Gs3, f);\
    \
    fp = _mm512_fmadd_pd(eta0, Gs1, r0);\
    fp = _mm512_fmadd_pd(zeta0, Gs2, fp);\
    \
    fpp = _mm512_mul_pd(eta0, Gs0);\
    fpp = _mm512_fmadd_pd(zeta0, Gs1, fpp);\
    \
    denom = _mm512_mul_pd(fp,fp);\
    denom = _mm512_mul_pd(denom,sixteen);\
    \
    denom = _mm512_fnmadd_pd(_mm512_mul_pd(f,fpp),twenty, denom);\
    /* not included: _mm512_abs_pd(denom) */;\
    denom = _mm512_sqrt_pd(denom);\
    denom = _mm512_add_pd(fp, denom);\
    \
    X = _mm512_fmsub_pd(X, denom, _mm512_mul_pd(f, five));\
    X = _mm512_div_pd(X, denom);

    // Initial guess
    __m512d dtr0i = _mm512_mul_pd(_dt,r0i);
    __m512d X = _mm512_mul_pd(dtr0i,eta0);
    X = _mm512_mul_pd(X,half);
    X = _mm512_fnmadd_pd(X,r0i,one);
    X = _mm512_mul_pd(dtr0i,X);

    // Iterations
    __m512d f, fp, fpp, denom, Gs0;
    HALLEY_STEP();
    HALLEY_STEP();
    NEWTON_STEP();
    // +1 below
    
    // Final Newton step (note: X not needed after this) 
    mm_stiefel_Gs13_avx512(&Gs1, &Gs2, &Gs3, beta, X);
    eta0Gs1zeta0Gs2 = _mm512_mul_pd(eta0, Gs1); 
    eta0Gs1zeta0Gs2 = _mm512_fmadd_pd(zeta0,Gs2, eta0Gs1zeta0Gs2); 
    ri = _mm512_add_pd(r0, eta0Gs1zeta0Gs2); 
    ri = _mm512_div_pd(one, ri); 

    // f and g function

    __m512d nf = _mm512_mul_pd(_M,Gs2); //negative f
    nf = _mm512_mul_pd(nf,r0i); 

    __m512d g = _mm512_fnmadd_pd(_M, Gs3, _dt);

    __m512d nfd = _mm512_mul_pd(_M, Gs1); // negative fd
    nfd = _mm512_mul_pd(nfd, r0i);
    nfd = _mm512_mul_pd(nfd, ri);

    __m512d ngd = _mm512_mul_pd(_M, Gs2); // negative gd
    ngd = _mm512_mul_pd(ngd, ri);

    __m512d nx = _mm512_fnmadd_pd(nf, p512->x, p512->x);
    nx = _mm512_fmadd_pd(g, p512->vx, nx);
    __m512d ny = _mm512_fnmadd_pd(nf, p512->y, p512->y);
    ny = _mm512_fmadd_pd(g, p512->vy, ny);
    __m512d nz = _mm512_fnmadd_pd(nf, p512->z, p512->z);
    nz = _mm512_fmadd_pd(g, p512->vz, nz);

    p512->vx = _mm512_fnmadd_pd(ngd, p512->vx, p512->vx);
    p512->vx = _mm512_fnmadd_pd(nfd, p512->x, p512->vx);
    p512->vy = _mm512_fnmadd_pd(ngd, p512->vy, p512->vy);
    p512->vy = _mm512_fnmadd_pd(nfd, p512->y, p512->vy);
    p512->vz = _mm512_fnmadd_pd(ngd, p512->vz, p512->vz);
    p512->vz = _mm512_fnmadd_pd(nfd, p512->z, p512->vz);

    p512->x = nx;
    p512->y = ny;
    p512->z = nz;
#ifdef PROF
    struct timeval time_end;
    gettimeofday(&time_end,NULL);
    walltime_kepler += time_end.tv_sec-time_beginning.tv_sec+(time_end.tv_usec-time_beginning.tv_usec)/1e6;
#endif
}

// Helper functions for the interaction step
static __m512d inline gravity_prefactor_avx512_one( __m512d dx, __m512d dy, __m512d dz) {
    __m512d r2 = _mm512_mul_pd(dx, dx);
    r2 = _mm512_fmadd_pd(dy,dy, r2);
    r2 = _mm512_fmadd_pd(dz,dz, r2);
    const __m512d r = _mm512_sqrt_pd(r2);
    const __m512d r3 = _mm512_mul_pd(r, r2);
    return _mm512_div_pd(one,r3); 
}

static __m512d inline gravity_prefactor_avx512( __m512d m, __m512d dx, __m512d dy, __m512d dz) {
    __m512d r2 = _mm512_mul_pd(dx, dx);
    r2 = _mm512_fmadd_pd(dy,dy, r2);
    r2 = _mm512_fmadd_pd(dz,dz, r2);
    const __m512d r = _mm512_sqrt_pd(r2);
    const __m512d r3 = _mm512_mul_pd(r, r2);
    return _mm512_div_pd(m,r3);
}

//Helper functions for combining seperate simulations into one
struct reb_simulation* combine_simulations(struct reb_simulation * r1, struct reb_simulation * r2, struct reb_simulation * r3, struct reb_simulation * r4){

    struct reb_simulation* r = reb_create_simulation();
    // Setup constants
    r->dt = r1->dt; // 5 days
    r->G = r1->G;
    r->exact_finish_time = r1->exact_finish_time;
    r->force_is_velocity_dependent = r1->force_is_velocity_dependent; 
    r->integrator = r1->integrator;
    r->ri_whfast512.gr_potential = r1->ri_whfast512.gr_potential;
    struct reb_particle p = {
            .m = r1->particles[0].m,
            .x = r1->particles[0].x, .y = r1->particles[0].y, .z = r1->particles[0].z,
            .vx = r1->particles[0].vx, .vy = r1->particles[0].vy, .vz = r1->particles[0].vz
    };
    reb_add(r, p);
    struct reb_particle p2 = {
            .m = r2->particles[0].m,
            .x = r2->particles[0].x, .y = r2->particles[0].y, .z = r2->particles[0].z,
            .vx = r2->particles[0].vx, .vy = r2->particles[0].vy, .vz = r2->particles[0].vz
    };
    reb_add(r, p2);
    struct reb_particle p3 = {
            .m = r3->particles[0].m,
            .x = r3->particles[0].x, .y = r3->particles[0].y, .z = r3->particles[0].z,
            .vx = r3->particles[0].vx, .vy = r3->particles[0].vy, .vz = r3->particles[0].vz
    };
    reb_add(r, p3);
    struct reb_particle p4 = {
            .m = r4->particles[0].m,
            .x = r4->particles[0].x, .y = r4->particles[0].y, .z = r4->particles[0].z,
            .vx = r4->particles[0].vx, .vy = r4->particles[0].vy, .vz = r4->particles[0].vz
    };
    reb_add(r, p4);
    for (int i = 1; i < 3; i++) {
        struct reb_particle p = {
            .m = r1->particles[i].m,
            .x = r1->particles[i].x, .y = r1->particles[i].y, .z = r1->particles[i].z,
            .vx = r1->particles[i].vx, .vy = r1->particles[i].vy, .vz = r1->particles[i].vz
        };
        reb_add(r, p);
    }
    for (int i = 1; i < 3; i++) {
        struct reb_particle p = {
            .m = r2->particles[i].m,
            .x = r2->particles[i].x, .y = r2->particles[i].y, .z = r2->particles[i].z,
            .vx = r2->particles[i].vx, .vy = r2->particles[i].vy, .vz = r2->particles[i].vz
        };
        reb_add(r, p);
    }
    for (int i = 1; i < 3; i++) {
        struct reb_particle p = {
            .m = r3->particles[i].m,
            .x = r3->particles[i].x, .y = r3->particles[i].y, .z = r3->particles[i].z,
            .vx = r3->particles[i].vx, .vy = r3->particles[i].vy, .vz = r3->particles[i].vz
        };
        reb_add(r, p);
    }
    for (int i = 1; i < 3; i++) {
        struct reb_particle p = {
            .m = r4->particles[i].m,
            .x = r4->particles[i].x, .y = r4->particles[i].y, .z = r4->particles[i].z,
            .vx = r4->particles[i].vx, .vy = r4->particles[i].vy, .vz = r4->particles[i].vz
        };
        reb_add(r, p);
    }
    return r;

}


// Performs one full interaction step
static void reb_whfast512_interaction_step(struct reb_particle_avx512* p, int gr_potential, double dt){
#ifdef PROF
    struct timeval time_beginning;
    gettimeofday(&time_beginning,NULL);
#endif
    struct reb_particle_avx512* restrict p_jh = p;
    
    __m512d x_j =  p_jh->x;
    __m512d y_j =  p_jh->y;
    __m512d z_j =  p_jh->z;
    __m512d dt512 = _mm512_set1_pd(dt); 

    // General relativistic corrections
    if (gr_potential){
        __m512d r2 = _mm512_mul_pd(x_j, x_j);
        r2 = _mm512_fmadd_pd(y_j, y_j, r2);
        r2 = _mm512_fmadd_pd(z_j, z_j, r2);
        const __m512d r4 = _mm512_mul_pd(r2, r2);
        __m512d prefac = _mm512_div_pd(gr_prefac,r4);
        prefac = _mm512_mul_pd(prefac, dt512);
        __m512d dvx = _mm512_mul_pd(prefac, x_j); 
        __m512d dvy = _mm512_mul_pd(prefac, y_j); 
        __m512d dvz = _mm512_mul_pd(prefac, z_j); 
        p_jh->vx  = _mm512_sub_pd(p_jh->vx, dvx);
        p_jh->vy  = _mm512_sub_pd(p_jh->vy, dvy);
        p_jh->vz  = _mm512_sub_pd(p_jh->vz, dvz);
       
        // Calculate back reaction onto star and apply them to planets (heliocentric) 
        dvx = _mm512_mul_pd(gr_prefac2, dvx); 
        dvy = _mm512_mul_pd(gr_prefac2, dvy); 
        dvz = _mm512_mul_pd(gr_prefac2, dvz); 
   
        dvx = _mm512_add_pd(_mm512_shuffle_pd(dvx, dvx, 0x55), dvx); // Swapping neighbouring elements
        //dvx = _mm512_add_pd(_mm512_permutex_pd(dvx, _MM_PERM_ABCD), dvx);
        dvy = _mm512_add_pd(_mm512_shuffle_pd(dvy, dvy, 0x55), dvy);
        //dvy = _mm512_add_pd(_mm512_permutex_pd(dvy, _MM_PERM_ABCD), dvy);
        dvz = _mm512_add_pd(_mm512_shuffle_pd(dvz, dvz, 0x55), dvz);
        //dvz = _mm512_add_pd(_mm512_permutex_pd(dvz, _MM_PERM_ABCD), dvz);
        
        p_jh->vx  = _mm512_sub_pd(p_jh->vx, dvx);
        p_jh->vy  = _mm512_sub_pd(p_jh->vy, dvy);
        p_jh->vz  = _mm512_sub_pd(p_jh->vz, dvz);
    }



    __m512d m_j = _mm512_mul_pd(p_jh->m, dt512);

    {  
        __m512d dx_j = _mm512_sub_pd(p_jh->x, _mm512_shuffle_pd(x_j, x_j, 0x55));
        __m512d dy_j = _mm512_sub_pd(p_jh->y, _mm512_shuffle_pd(y_j, y_j, 0x55));
        __m512d dz_j = _mm512_sub_pd(p_jh->z, _mm512_shuffle_pd(z_j, z_j, 0x55));
        m_j = _mm512_shuffle_pd(m_j, m_j, 0x55);

        __m512d prefact = gravity_prefactor_avx512_one(dx_j, dy_j, dz_j);

         __m512d prefact1 = _mm512_mul_pd(prefact, m_j);
        p_jh->vx = _mm512_fnmadd_pd(prefact1, dx_j, p_jh->vx); 
        p_jh->vy = _mm512_fnmadd_pd(prefact1, dy_j, p_jh->vy); 
        p_jh->vz = _mm512_fnmadd_pd(prefact1, dz_j, p_jh->vz); 
    }
    

#ifdef PROF
    struct timeval time_end;
    gettimeofday(&time_end,NULL);
    walltime_interaction += time_end.tv_sec-time_beginning.tv_sec+(time_end.tv_usec-time_beginning.tv_usec)/1e6;
#endif
}

// Convert inertial coordinates to democratic heliocentric coordinates
// Note: this is only called at the beginning. Speed is not a concern.
static void inertial_to_democraticheliocentric_posvel(struct reb_simulation* r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    struct reb_particle_avx512* p512 = aligned_alloc(64,sizeof(struct reb_particle_avx512));
    struct reb_particle* particles = r->particles;
    struct reb_particle_avx512* p_jh = ri_whfast512->p_jh;
    const unsigned int N = r->N;
    double val[8];
#define CONVERT2AVX(x, p) \
    for (unsigned int i=4;i<N;i++){\
        val[i-4] = particles[i].x;\
    }\
    for (unsigned int i=N;i<12;i++){\
        if (p){\
            val[i-4] = 100+i;\
        }else{\
            val[i-4] = 0.0;\
        }\
    }\
    p512->x = _mm512_loadu_pd(&val);

    CONVERT2AVX(m,0);
    CONVERT2AVX(x, 1);
    CONVERT2AVX(y, 1);
    CONVERT2AVX(z, 1);
    CONVERT2AVX(vx, 0);
    CONVERT2AVX(vy, 0);
    CONVERT2AVX(vz, 0 );

    p_jh->m = p512->m;
    double tempstorage[8];
    __m512d temp = _mm512_add_pd(_mm512_shuffle_pd(p512->m, p512->m, 0x55), p512->m);
    _mm512_storeu_pd(&tempstorage, temp);

    double mtot = tempstorage[0] + particles[0].m;
    double mtot2 = tempstorage[2] + particles[1].m;
    double mtot3 = tempstorage[4] + particles[2].m;
    double mtot4 = tempstorage[6] + particles[3].m;
    ri_whfast512->p_jh0.m = mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.m = mtot2;
    ri_whfast512->p_jh03.m = mtot3;
    ri_whfast512->p_jh04.m = mtot4;

    __m512d xm = _mm512_mul_pd(p512->x,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(xm, xm, 0x55), xm);
    _mm512_storeu_pd(&tempstorage, temp);
    double x0 = tempstorage[0] + particles[0].m*particles[0].x;
    double x02 = tempstorage[2] + particles[1].m*particles[1].x;
    double x03 = tempstorage[4] + particles[2].m*particles[2].x;
    double x04 = tempstorage[6] + particles[3].m*particles[3].x;
    ri_whfast512->p_jh0.x = x0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.x = x02/mtot2;
    ri_whfast512->p_jh03.x = x03/mtot3;
    ri_whfast512->p_jh04.x = x04/mtot4;

    __m512d ym = _mm512_mul_pd(p512->y,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(ym, ym, 0x55), ym);
    _mm512_storeu_pd(&tempstorage, temp);
    double y0 = tempstorage[0] + particles[0].m*particles[0].y;
    double y02 = tempstorage[2] + particles[1].m*particles[1].y;
    double y03 = tempstorage[4] + particles[2].m*particles[2].y;
    double y04 = tempstorage[6] + particles[3].m*particles[3].y;
    ri_whfast512->p_jh0.y = y0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.y = y02/mtot2;
    ri_whfast512->p_jh03.y = y03/mtot3;
    ri_whfast512->p_jh04.y = y04/mtot4;

    __m512d zm = _mm512_mul_pd(p512->z,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(zm, zm, 0x55), zm);
    _mm512_storeu_pd(&tempstorage, temp);
    double z0 = tempstorage[0] + particles[0].m*particles[0].z;
    double z02 = tempstorage[2] + particles[1].m*particles[1].z;
    double z03 = tempstorage[4] + particles[2].m*particles[2].z;
    double z04 = tempstorage[6] + particles[3].m*particles[3].z;
    ri_whfast512->p_jh0.z = z0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.z = z02/mtot2;
    ri_whfast512->p_jh03.z = z03/mtot3;
    ri_whfast512->p_jh04.z = z04/mtot4;
    
    __m512d vxm = _mm512_mul_pd(p512->vx,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vxm, vxm, 0x55), vxm);
    _mm512_storeu_pd(&tempstorage, temp);
    double vx0 = tempstorage[0] + particles[0].m*particles[0].vx;
    double vx02 = tempstorage[2] + particles[1].m*particles[1].vx;
    double vx03 = tempstorage[4] + particles[2].m*particles[2].vx;
    double vx04 = tempstorage[6] + particles[3].m*particles[3].vx;
    ri_whfast512->p_jh0.vx = vx0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.vx = vx02/mtot2;
    ri_whfast512->p_jh03.vx = vx03/mtot3;
    ri_whfast512->p_jh04.vx = vx04/mtot4;

    __m512d vym = _mm512_mul_pd(p512->vy,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vym, vym, 0x55), vym);
    _mm512_storeu_pd(&tempstorage, temp);
    double vy0 = tempstorage[0] + particles[0].m*particles[0].vy;
    double vy02 = tempstorage[2] + particles[1].m*particles[1].vy;
    double vy03 = tempstorage[4] + particles[2].m*particles[2].vy;
    double vy04 = tempstorage[6] + particles[3].m*particles[3].vy;
    ri_whfast512->p_jh0.vy = vy0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.vy = vy02/mtot2;
    ri_whfast512->p_jh03.vy = vy03/mtot3;
    ri_whfast512->p_jh04.vy = vy04/mtot4;

    __m512d vzm = _mm512_mul_pd(p512->vz,p512->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vzm, vzm, 0x55), vzm);
    _mm512_storeu_pd(&tempstorage, temp);
    double vz0 = tempstorage[0] + particles[0].m*particles[0].vz;
    double vz02 = tempstorage[2] + particles[1].m*particles[1].vz;
    double vz03 = tempstorage[4] + particles[2].m*particles[2].vz;
    double vz04 = tempstorage[6] + particles[3].m*particles[3].vz;
    ri_whfast512->p_jh0.vz = vz0/mtot; //CHANGE mtot, change rduce add functions
    ri_whfast512->p_jh02.vz = vz02/mtot2;
    ri_whfast512->p_jh03.vz = vz03/mtot3;
    ri_whfast512->p_jh04.vz = vz04/mtot4;

    double x1 = particles[1].x;
    double x = particles[0].x;
    double x2 = particles[2].x;
    double x3 = particles[3].x;
    
    xm = _mm512_set_pd(x3, x3, x2, x2, x1, x1, x, x);
    p_jh->x = _mm512_sub_pd(p512->x, xm);

    double y1 = particles[1].y;
    double y = particles[0].y;
    double y2 = particles[2].y;
    double y3 = particles[3].y;
    ym = _mm512_set_pd(y3, y3, y2, y2, y1, y1, y, y);
    p_jh->y = _mm512_sub_pd(p512->y, ym);

    double z1 = particles[1].z;
    double z = particles[0].z;
    double z2 = particles[2].z;
    double z3 = particles[3].z;
    zm = _mm512_set_pd(z3, z3, z2, z2, z1, z1, z, z);
    p_jh->z = _mm512_sub_pd(p512->z, zm);

    
    vxm = _mm512_set_pd(vx04, vx04, vx03, vx03, vx02, vx02, vx0, vx0);
    p_jh->vx = _mm512_sub_pd(p512->vx, vxm);
    vym = _mm512_set_pd(vy04, vy04, vy03, vy03, vy02, vy02, vy0, vy0);
    p_jh->vy = _mm512_sub_pd(p512->vy, vym);
    vzm = _mm512_set_pd(vz04, vz04, vz03, vz03, vz02, vz02, vz0, vz0);
    p_jh->vz = _mm512_sub_pd(p512->vz, vzm);
     
}
// Convert democratic heliocentric coordinates to inertial coordinates
// Note: this is only called at the end. Speed is not a concern.
static void democraticheliocentric_to_inertial_posvel(struct reb_simulation* r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    struct reb_particle* particles = r->particles;
    struct reb_particle_avx512* p512 = aligned_alloc(64,sizeof(struct reb_particle_avx512));
    struct reb_particle_avx512* p_jh = ri_whfast512->p_jh;
    const double mtot = ri_whfast512->p_jh0.m;
    const double mtot2 = ri_whfast512->p_jh02.m; 
    const double mtot3 = ri_whfast512->p_jh03.m;
    const double mtot4 = ri_whfast512->p_jh04.m;  
    const unsigned int N = r->N;

    double tempstorage[8];

    __m512d x0 = _mm512_mul_pd(p_jh->x,p_jh->m);//double M = _mm512_reduce_add_pd(p_jh->m);
    __m512d temp = _mm512_add_pd(_mm512_shuffle_pd(x0, x0, 0x55), x0);
    _mm512_storeu_pd(&tempstorage, temp);
    double x0s = tempstorage[0]/mtot;
    double x0s2 = tempstorage[2]/mtot2;
    double x0s3 = tempstorage[4]/mtot3;
    double x0s4 = tempstorage[6]/mtot4;

    __m512d y0 = _mm512_mul_pd(p_jh->y,p_jh->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(y0, y0, 0x55), y0);
    _mm512_storeu_pd(&tempstorage, temp);
    double y0s = tempstorage[0]/mtot;
    double y0s2 = tempstorage[2]/mtot2;
    double y0s3 = tempstorage[4]/mtot3;
    double y0s4 = tempstorage[6]/mtot4;

    __m512d z0 = _mm512_mul_pd(p_jh->z,p_jh->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(z0, z0, 0x55), z0);
    _mm512_storeu_pd(&tempstorage, temp);
    double z0s = tempstorage[0]/mtot;
    double z0s2 = tempstorage[2]/mtot2;
    double z0s3 = tempstorage[4]/mtot3;
    double z0s4 = tempstorage[6]/mtot4;

    particles[0].x  = ri_whfast512->p_jh0.x - x0s;
    particles[0].y  = ri_whfast512->p_jh0.y - y0s;
    particles[0].z  = ri_whfast512->p_jh0.z - z0s;

    particles[1].x  = ri_whfast512->p_jh02.x - x0s2;
    particles[1].y  = ri_whfast512->p_jh02.y - y0s2;
    particles[1].z  = ri_whfast512->p_jh02.z - z0s2;

    particles[2].x  = ri_whfast512->p_jh03.x - x0s3;
    particles[2].y  = ri_whfast512->p_jh03.y - y0s3;
    particles[2].z  = ri_whfast512->p_jh03.z - z0s3;

    particles[3].x  = ri_whfast512->p_jh04.x - x0s4;
    particles[3].y  = ri_whfast512->p_jh04.y - y0s4;
    particles[3].z  = ri_whfast512->p_jh04.z - z0s4;

    double x = particles[0].x;
    double x1 = particles[1].x;
    double x2 = particles[2].x;
    double x3 = particles[3].x;
    x0 = _mm512_set_pd(x3, x3, x2, x2, x1, x1, x, x);
    p512->x = _mm512_add_pd(p_jh->x, x0);

    double y = particles[0].y;
    double y1 = particles[1].y;
    double y2 = particles[2].y;
    double y3 = particles[3].y;
    y0 = _mm512_set_pd(y3, y3, y2, y2, y1, y1, y, y);
    p512->y = _mm512_add_pd(p_jh->y, y0);

    double z = particles[0].z;
    double z1 = particles[1].z;
    double z2 = particles[2].z;
    double z3 = particles[3].z;
    z0 = _mm512_set_pd(z3, z3, z2, z2, z1, z1, z, z);
    p512->z = _mm512_add_pd(p_jh->z, z0);

    double vx = ri_whfast512->p_jh0.vx;
    double vx2 = ri_whfast512->p_jh02.vx;
    double vx3 = ri_whfast512->p_jh03.vx;
    double vx4 = ri_whfast512->p_jh04.vx;
    __m512d vx0 = _mm512_set_pd(vx4, vx4, vx3, vx3, vx2, vx2, vx, vx);
    p512->vx = _mm512_add_pd(p_jh->vx, vx0);

    double vy = ri_whfast512->p_jh0.vy;
    double vy2 = ri_whfast512->p_jh02.vy;
    double vy3 = ri_whfast512->p_jh03.vy;
    double vy4 = ri_whfast512->p_jh04.vy;
    __m512d vy0 = _mm512_set_pd(vy4, vy4, vy3, vy3, vy2, vy2, vy, vy);
    p512->vy = _mm512_add_pd(p_jh->vy, vy0);

    double vz = ri_whfast512->p_jh0.vz;
    double vz2 = ri_whfast512->p_jh02.vz;
    double vz3 = ri_whfast512->p_jh03.vz;
    double vz4 = ri_whfast512->p_jh04.vz;
    __m512d vz0 = _mm512_set_pd(vz4, vz4, vz3, vz3, vz2, vz2, vz, vz);
    p512->vz = _mm512_add_pd(p_jh->vz, vz0);

    const double m02 = particles[1].m;
    const double m0 = particles[0].m;
    const double m03 = particles[2].m;
    const double m04 = particles[3].m;

    
    vx0 = _mm512_mul_pd(p_jh->vx, p_jh->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vx0, vx0, 0x55), vx0);
    _mm512_storeu_pd(&tempstorage, temp);
    double vx0s = tempstorage[0]/m0;
    double vx0s2 = tempstorage[2]/m02;
    double vx0s3 = tempstorage[4]/m03;
    double vx0s4 = tempstorage[6]/m04;

    vy0 = _mm512_mul_pd(p_jh->vy, p_jh->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vy0, vy0, 0x55), vy0);
    _mm512_storeu_pd(&tempstorage, temp);
    double vy0s = tempstorage[0]/m0;
    double vy0s2 = tempstorage[2]/m02;
    double vy0s3 = tempstorage[4]/m03;
    double vy0s4 = tempstorage[6]/m04;
    
    vz0 = _mm512_mul_pd(p_jh->vz, p_jh->m);
    temp = _mm512_add_pd(_mm512_shuffle_pd(vz0, vz0, 0x55), vz0);
    _mm512_storeu_pd(&tempstorage, temp);
    double vz0s = tempstorage[0]/m0;
    double vz0s2 = tempstorage[2]/m02;
    double vz0s3 = tempstorage[4]/m03;
    double vz0s4 = tempstorage[6]/m04;

    particles[3].vx = ri_whfast512->p_jh04.vx -vx0s4;
    particles[3].vy = ri_whfast512->p_jh04.vy -vy0s4;
    particles[3].vz = ri_whfast512->p_jh04.vz -vz0s4;

    particles[2].vx = ri_whfast512->p_jh03.vx -vx0s3;
    particles[2].vy = ri_whfast512->p_jh03.vy -vy0s3;
    particles[2].vz = ri_whfast512->p_jh03.vz -vz0s3;

    particles[1].vx = ri_whfast512->p_jh02.vx -vx0s2;
    particles[1].vy = ri_whfast512->p_jh02.vy -vy0s2;
    particles[1].vz = ri_whfast512->p_jh02.vz -vz0s2;

    particles[0].vx = ri_whfast512->p_jh0.vx -vx0s;
    particles[0].vy = ri_whfast512->p_jh0.vy -vy0s;
    particles[0].vz = ri_whfast512->p_jh0.vz -vz0s;

    // Only called at the end. Speed is not a concern.

    double val[8];
#define CONVERT2PAR(x) \
    _mm512_storeu_pd(&val, p512->x);\
    for (unsigned int i=4;i<N;i++){\
        particles[i].x = val[i-4];\
    }
   
    CONVERT2PAR(x); 
    CONVERT2PAR(y); 
    CONVERT2PAR(z); 
    CONVERT2PAR(vx); 
    CONVERT2PAR(vy); 
    CONVERT2PAR(vz); 
    free(p512);
}

// Performs one complete jump step
static void reb_whfast512_jump_step(struct reb_particle_avx512 * p_jh, __m512d m0, const double _dt){
#ifdef PROF
    struct timeval time_beginning;
    gettimeofday(&time_beginning,NULL);
#endif

    __m512d ts = _mm512_set1_pd(_dt);
    
    __m512d pf512 = _mm512_div_pd(ts, m0);


    __m512d sumx = _mm512_mul_pd(p_jh->m, p_jh->vx);
    __m512d sumy = _mm512_mul_pd(p_jh->m, p_jh->vy);
    __m512d sumz = _mm512_mul_pd(p_jh->m, p_jh->vz);


    sumx = _mm512_add_pd(_mm512_shuffle_pd(sumx, sumx, 0x55), sumx); // Swapping neighbouring elements
    //sumx = _mm512_add_pd(_mm512_permutex_pd(sumx, _MM_PERM_ABCD), sumx);

    sumy = _mm512_add_pd(_mm512_shuffle_pd(sumy, sumy, 0x55), sumy);
    //sumy = _mm512_add_pd(_mm512_permutex_pd(sumy, _MM_PERM_ABCD), sumy);

    sumz = _mm512_add_pd(_mm512_shuffle_pd(sumz, sumz, 0x55), sumz);
    //sumz = _mm512_add_pd(_mm512_permutex_pd(sumz, _MM_PERM_ABCD), sumz);

    p_jh->x = _mm512_fmadd_pd(sumx, pf512, p_jh->x); 
    p_jh->y = _mm512_fmadd_pd(sumy, pf512, p_jh->y); 
    p_jh->z = _mm512_fmadd_pd(sumz, pf512, p_jh->z); 


#ifdef PROF
    struct timeval time_end;
    gettimeofday(&time_end,NULL);
    walltime_jump += time_end.tv_sec-time_beginning.tv_sec+(time_end.tv_usec-time_beginning.tv_usec)/1e6;
#endif
}

// Performs one full centre of mass step (H_0)
static void reb_whfast512_com_step(struct reb_simulation* r, const double _dt){
#ifdef PROF
    struct timeval time_beginning;
    gettimeofday(&time_beginning,NULL);
#endif
    r->ri_whfast512.p_jh0.x += _dt*r->ri_whfast512.p_jh0.vx;
    r->ri_whfast512.p_jh0.y += _dt*r->ri_whfast512.p_jh0.vy;
    r->ri_whfast512.p_jh0.z += _dt*r->ri_whfast512.p_jh0.vz;
    r->ri_whfast512.p_jh02.x += _dt*r->ri_whfast512.p_jh02.vx;
    r->ri_whfast512.p_jh02.y += _dt*r->ri_whfast512.p_jh02.vy;
    r->ri_whfast512.p_jh02.z += _dt*r->ri_whfast512.p_jh02.vz;
    r->ri_whfast512.p_jh03.x += _dt*r->ri_whfast512.p_jh03.vx;
    r->ri_whfast512.p_jh03.y += _dt*r->ri_whfast512.p_jh03.vy;
    r->ri_whfast512.p_jh03.z += _dt*r->ri_whfast512.p_jh03.vz;
    r->ri_whfast512.p_jh04.x += _dt*r->ri_whfast512.p_jh04.vx;
    r->ri_whfast512.p_jh04.y += _dt*r->ri_whfast512.p_jh04.vy;
    r->ri_whfast512.p_jh04.z += _dt*r->ri_whfast512.p_jh04.vz;
#ifdef PROF
    struct timeval time_end;
    gettimeofday(&time_end,NULL);
    walltime_com += time_end.tv_sec-time_beginning.tv_sec+(time_end.tv_usec-time_beginning.tv_usec)/1e6;
#endif
}

void static recalculate_constants(struct reb_simulation* r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    const unsigned int N = r->N;
    half = _mm512_set1_pd(0.5); 
    one = _mm512_add_pd(half, half); 
    two = _mm512_add_pd(one, one); 
    five = _mm512_set1_pd(5.); 
    sixteen = _mm512_set1_pd(16.); 
    twenty = _mm512_set1_pd(20.); 
    double m0 = r->particles[0].m;
    double m1 = r->particles[1].m;
    double m2 = r->particles[2].m;
    double m3 = r->particles[3].m;
    _M = _mm512_set_pd(m3, m3, m2, m2, m1, m1, m0, m0); 
    //so1 = _mm512_set_epi64(1,2,3,0,6,7,4,5);
    //so2 = _mm512_set_epi64(3,2,1,0,6,5,4,7);
    for(unsigned int i=0;i<35;i++){
        invfactorial512[i] = _mm512_set1_pd(invfactorial[i]); 
    }

    // GR prefactors. Note: assumes units of AU, year/2pi.
    double c = 10065.32;
    double gr0 = 6.*m0*m0/(c*c);
    double gr1 = 6.*m1*m1/(c*c);
    double gr2 = 6.*m2*m2/(c*c);
    double gr3 = 6.*m3*m3/(c*c);
    gr_prefac = _mm512_set_pd(gr3, gr3, gr2, gr2, gr1, gr1, gr0, gr0);
    double _gr_prefac2[8];
    for(unsigned int i=4;i<6;i++){
        _gr_prefac2[i-4] = r->particles[i].m/m0;
    }
    for (unsigned int i =6; i<8; i++){
        _gr_prefac2[i-4] = r->particles[i].m/m1;
    }
    for (unsigned int i =8; i<10; i++){
        _gr_prefac2[i-4] = r->particles[i].m/m2;
    }
    for (unsigned int i =10; i<12; i++){
        _gr_prefac2[i-4] = r->particles[i].m/m3;
    }
    for(unsigned int i=N;i<12;i++){
        _gr_prefac2[i-4] = 0;
    }
    gr_prefac2 = _mm512_loadu_pd(&_gr_prefac2);
    ri_whfast512->recalculate_constants = 0;

}

// Main integration routine
void reb_integrator_whfast512_part1(struct reb_simulation* const r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    const double dt = r->dt;
    
    if (ri_whfast512->allocated_N==0){
        // Check if all assumptions are satisfied.
        // Note: These are not checked every timestep. 
        // So it is possible for the user to screw things up.
        if (r->dt<=0.0 ){
            reb_error(r, "WHFast512 does not support negative timesteps. To integrate backwards, flip the sign of the velocities.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if (r->N_var!=0){
            reb_error(r, "WHFast512 does not support variational particles.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if (r->exact_finish_time!=0 ){
            reb_error(r, "WHFast512 requires exact_finish_time=0.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if (r->N>12 ){
            reb_error(r, "WHFast512 supports a maximum of 10 particles.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if (r->G!=1.0 ){
            reb_error(r, "WHFast512 requires units in which G=1. Please rescale your system.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if ((r->N_active!=-1 && r->N_active!=r->N) ){
            reb_error(r, "WHFast512 does not support test particles.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        ri_whfast512->p_jh = aligned_alloc(64,sizeof(struct reb_particle_avx512));
        if (!ri_whfast512->p_jh ){
            reb_error(r, "WHFast512 was not able to allocate memory.");
            r->status = REB_EXIT_ERROR;
            return;
        }
        if (r->exit_min_distance || r->exit_max_distance ){
            reb_warning(r, "You are using WHFast512 together with the flags exit_min_distance and/or exit_max_distance. With the current implementation, these flags will only check the last synchronized positions. In addition they might slow down WHFast512 significantly. If you need to use these flags, please open an issue on GitHub for further advice.");
        }
        ri_whfast512->allocated_N=1;
        ri_whfast512->recalculate_constants = 1;
        r->gravity = REB_GRAVITY_NONE; // WHFast512 uses its own gravity routine.
    }

    if (ri_whfast512->recalculate_constants){
        recalculate_constants(r);
    } 

    if (ri_whfast512->is_synchronized){
        inertial_to_democraticheliocentric_posvel(r);
    }

    struct reb_particle_avx512* p = ri_whfast512->p_jh;

    if (ri_whfast512->is_synchronized){
        // First half DRIFT step
        reb_whfast512_kepler_step(p, dt/2.);    
        reb_whfast512_com_step(r, dt/2.);
    }else{
        // Combined DRIFT step
        reb_whfast512_kepler_step(p, dt); // full timestep
        reb_whfast512_com_step(r, dt);
    }

    if (ri_whfast512->gr_potential ){
        reb_whfast512_jump_step(p, _M, dt/2.);
    }else{
        reb_whfast512_jump_step(p, _M, dt);
        
    }

    reb_whfast512_interaction_step(p, ri_whfast512->gr_potential, dt);; 
    
    if (ri_whfast512->gr_potential){
        reb_whfast512_jump_step(p, _M, dt/2.);
    }

    ri_whfast512->is_synchronized = 0;
    
    r->t += dt;
    r->dt_last_done = dt;
}

#else // AVX512
// Dummy function when AVX512 is not available
void reb_integrator_whfast512_part1(struct reb_simulation* const r){
    reb_error(r, "WHFast512 is not available. Please make sure your CPU supports AVX512 instructions, then recompile REBOUND with the AVX512 option turned on in the Makefile or set the AVX512 environment variable to 1 before running pip install.");
    r->status = REB_EXIT_ERROR;
}
#endif // AVX512

// Synchronization routine. Called every time an output is needed.
void reb_integrator_whfast512_synchronize(struct reb_simulation* const r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    if (ri_whfast512->is_synchronized == 0){
#ifdef AVX512
        struct reb_particle_avx512* sync_pj = NULL;
        struct reb_particle sync_pj0 = {0};
        struct reb_particle sync_pj02 = {0};
        struct reb_particle sync_pj03 = {0};
        struct reb_particle sync_pj04 = {0};
        if (ri_whfast512->recalculate_constants){ 
            // Needed if no step has ever been done before (like SA)
            recalculate_constants(r);
        } 
        if (ri_whfast512->keep_unsynchronized){
            sync_pj = aligned_alloc(64,sizeof(struct reb_particle_avx512));
            memcpy(sync_pj,ri_whfast512->p_jh, sizeof(struct reb_particle_avx512));
            sync_pj0 = ri_whfast512->p_jh0;
            sync_pj02 = ri_whfast512->p_jh02;
            sync_pj03 = ri_whfast512->p_jh03;
            sync_pj04 = ri_whfast512->p_jh04;
        }
        reb_whfast512_kepler_step(ri_whfast512->p_jh, r->dt/2.);    
        reb_whfast512_com_step(r, r->dt/2.);
        democraticheliocentric_to_inertial_posvel(r);
        if (ri_whfast512->keep_unsynchronized){
            memcpy(ri_whfast512->p_jh, sync_pj, sizeof(struct reb_particle_avx512));
            ri_whfast512->p_jh0 = sync_pj0;
            ri_whfast512->p_jh02 = sync_pj02;
            ri_whfast512->p_jh03 = sync_pj03;
            ri_whfast512->p_jh04 = sync_pj04;
            free(sync_pj);
        }else{
            ri_whfast512->is_synchronized = 1;
        }
#else // No AVX512 available
      // Using WHFast as a workaround.
      // Not bit-wise reproducible. 
        struct reb_simulation_integrator_whfast* const ri_whfast = &(r->ri_whfast);
        reb_warning(r, "WHFast512 is not available. Synchronization is provided using WHFast and is not bit-compatible to WHFast512.");
        reb_integrator_whfast_init(r);
        ri_whfast->p_jh[0] = ri_whfast512->p_jh0;
        for (unsigned int i=1;i<r->N;i++){
            ri_whfast->p_jh[i].m = ri_whfast512->p_jh->m[i-1];
            ri_whfast->p_jh[i].x = ri_whfast512->p_jh->x[i-1];
            ri_whfast->p_jh[i].y = ri_whfast512->p_jh->y[i-1];
            ri_whfast->p_jh[i].z = ri_whfast512->p_jh->z[i-1];
            ri_whfast->p_jh[i].vx = ri_whfast512->p_jh->vx[i-1];
            ri_whfast->p_jh[i].vy = ri_whfast512->p_jh->vy[i-1];
            ri_whfast->p_jh[i].vz = ri_whfast512->p_jh->vz[i-1];
        }
        ri_whfast->coordinates = REB_WHFAST_COORDINATES_DEMOCRATICHELIOCENTRIC;
        ri_whfast->is_synchronized = 0;
        reb_integrator_whfast_synchronize(r);
        
        ri_whfast512->is_synchronized = ri_whfast->is_synchronized;
#endif // AVX512
    }
}
//LEAVE FOR NOW
// Free memory and reset all constants.
// This needs to be called when the timestep, the number of particles, masses, etc are changed, 
void reb_integrator_whfast512_reset(struct reb_simulation* const r){
    struct reb_simulation_integrator_whfast512* const ri_whfast512 = &(r->ri_whfast512);
    if (ri_whfast512->allocated_N){
        free(ri_whfast512->p_jh);
    }
    ri_whfast512->p_jh = NULL;
    ri_whfast512->allocated_N = 0;
    ri_whfast512->gr_potential = 0;
    ri_whfast512->is_synchronized = 1;
    ri_whfast512->keep_unsynchronized = 0;
    ri_whfast512->recalculate_constants = 1;
}

// Everything is in part 1 for this integrator
void reb_integrator_whfast512_part2(struct reb_simulation* const r){
}
