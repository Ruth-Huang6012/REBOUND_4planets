#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "main.h"
#include "particle.h"
#include "boundaries.h"

void problem_init(int argc, char* argv[]){
	// Setup constants
	dt = 1e-3;
	boxsize_x = 3;
	boxsize_y = 3;
	boxsize_z = 3;
	// Setup particle structures
	init_particles(2);
	// Initial conditions
	/*
	particles[0].x  = 0;
	particles[0].y  = 0;
	particles[0].z  = 0;
	particles[0].vx = 0;
	particles[0].vy = 0;
	particles[0].vz = 0;
	particles[0].ax = 0;
	particles[0].ay = 0;
	particles[0].az = 0;
	particles[0].m  = 1;
	particles[1].x  = 1;
	particles[1].y  = 0;
	particles[1].z  = 0;
	particles[1].vx = 0;
	particles[1].vy = 1;
	particles[1].vz = 0;
	particles[1].ax = 0;
	particles[1].ay = 0;
	particles[1].az = 0;
	particles[1].m  = 0;
	*/
	particles[0].x  = -0.5;
	particles[0].y  = 0;
	particles[0].z  = 0;
	particles[0].vx = 0;
	particles[0].vy = -1;
	particles[0].vz = 0;
	particles[0].ax = 0;
	particles[0].ay = 0;
	particles[0].az = 0;
	particles[0].m  = 2;
	particles[1].x  = 0.5;
	particles[1].y  = 0;
	particles[1].z  = 0;
	particles[1].vx = 0;
	particles[1].vy = 1;
	particles[1].vz = 0;
	particles[1].ax = 0;
	particles[1].ay = 0;
	particles[1].az = 0;
	particles[1].m  = 2;
	// Do not use any ghost boxes
	nghostx = 0;
	nghosty = 0;
	nghostz = 0;
}

void problem_inloop(){

}

void problem_output(){

}

void problem_finish(){

}