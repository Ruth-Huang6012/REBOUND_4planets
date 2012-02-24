/**
 * @file 	output.h
 * @brief 	Output routines.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @details	If MPI is enabled, most functions output one file per
 * node. They automatically add a subscript _0, _1, .. to each file.
 * The user has to join them together manually. One exception is 
 * output_append_velocity_dispersion() which only outputs one file.
 * 
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
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
#ifndef _OUTPUT_H
#define _OUTPUT_H

/**
 * This function checks if a new output is required at this time.
 * @return The return value is 1 if an output is required and 0 otherwise.
 * @param interval Output interval.
 */
int output_check(double interval);

/**
 * This function checks if a new output is required at this time.
 * @return The return value is 1 if an output is required and 0 otherwise.
 * @param interval Output interval.
 * @param phase Phase (if 0, then this function is equal to output_check()).
 */
int output_check_phase(double interval,double phase);

/**
 * Outputs the current number of particles, the time and the time difference since the last output to the screen.
 */

void output_timing();
/**
 * Outputs an ASCII file with the positions and velocities of all particles.
 * @param filename Output filename.
 */
void output_ascii(char* filename);

/**
 * Outputs an ASCII file with orbital paramters of all particles.
 * @details The orbital parameters are calculated with respect to (x,y,z)=(0,0,0) 
 * and assume a stellar mass of 1.
 * @param filename Output filename.
 */
void output_orbits(char* filename);

/**
 * Appends an ASCII file with orbital paramters of all particles.
 * @details The orbital parameters are calculated with respect to (x,y,z)=(0,0,0) 
 * and assume a stellar mass of 1.
 * @param filename Output filename.
 */
void output_append_orbits(char* filename);

/**
 * Appends the positions and velocities of all particles to an ASCII file.
 * @param filename Output filename.
 */
void output_append_ascii(char* filename);

/**
 * Dumps all particle structs into a binary file.
 * @param filename Output filename.
 */
void output_binary(char* filename);

/**
 * Dumps only the positions of all particles into a binary file.
 * @param filename Output filename.
 */
void output_binary_positions(char* filename);

/**
 * Appends the velocity dispersion of the particles to an ASCII file.
 * @param filename Output filename.
 */
void output_append_velocity_dispersion(char* filename);

#if defined(OPENGL) && defined(LIBPNG)
/**
 * Outputs a screenshot of the current OpenGL view.
 * @details Requires OpenGL and LIBPNG to be installed.
 * The filename is generated by appending a nine digit number and ".png" to dirname. 
 * The number is increased every time an output is generated. One can then use ffmpeg to 
 * compile movies with the command 'ffmpeg -qscale 5 -r 20 -b 9600 -i %09d.png movie.mp4'.
 * @param dirname Output directory (e.g. 'src/'). Must end with a slash. Directory must exist.
 */
void output_png(char* dirname);
/**
 * Outputs a single screenshot of the current OpenGL view.
 * @details Requires OpenGL and LIBPNG to be installed.
 * @param filename Output filename.
 */
void output_png_single(char* filename);
#endif

#endif
