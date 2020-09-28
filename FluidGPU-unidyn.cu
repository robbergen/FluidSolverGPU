#include "FluidGPU-unidyn.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

float kernel(float r) {
	if (r >= 0 && r <= cutoff) {
		return 1. / 3.14159 / (powf(cutoff, 3))*(1 - 3. / 2. * powf((r / cutoff), 2) + 3. / 4. * powf((r / cutoff), 3));
	}
	else if (r > cutoff && r < (2 * cutoff)) {
		return 1. / 3.14159 / (powf(cutoff, 3)) * 1 / 4. * powf(2 - (r / cutoff), 3);
	}
	else {
		return 0;
	}
}

float kernel_test(float r) {
	if (r >= 0 && r <= cutoff) {
		return 1. / 3.14159 / (powf(cutoff, 4))*(1 - 3. * powf((r / cutoff), 1) + 9. / 4. * powf((r / cutoff), 2));
	}
	else if (r > cutoff && r < (2 * cutoff)) {
		return -1. / 3.14159 / (powf(cutoff, 4)) * 1 / 2. * powf(2 - (r / cutoff), 2);
	}
	else {
		return 0;
	}
}

float kernel_derivative(float r) {
	if (r < cutoff) {
		return -45.0 / 3.14159 / powf(cutoff, 6)*powf((cutoff - r), 2);
	}
	else {
		return 0;
	}

}

//Dot product
inline float dot_prod(float x1, float y1, float z1, float x2, float y2, float z2) {
	return x1*x2 + y1*y2 + z1*z2;
}

//Cross products
inline float cross_prod_x(float x1, float y1, float z1, float x2, float y2, float z2) {
	return y1*z2 - z1*y2;
}

inline float cross_prod_y(float x1, float y1, float z1, float x2, float y2, float z2) {
	return -x1*z2 + z1*x2;
}

inline float cross_prod_z(float x1, float y1, float z1, float x2, float y2, float z2) {
	return x1*y2 - y1*x2;
}

__device__ int morton(unsigned int x, unsigned int y, unsigned int z) {
	//int x = (bidx / GRIDSIZE / GRIDSIZE);
	//int y = (bidx / GRIDSIZE % GRIDSIZE);
	//int z = (bidx % GRIDSIZE);

	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;

	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;

	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;

	return x | (y << 1) | (z << 2);

}

__device__ inline int demorton(unsigned int x, int b) {
	//b should be 0 for x, 1 for y, 2 for z
	switch (b) {
	case 0: break;
	case 1: x = (x >> 1);
		break;
	case 2: x = (x >> 2);
		break;
	}
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x | (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}



__global__ void findneighbours(int *cell, int *start, int *start_copy, int *end, int nspts, int x) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx<1) {printf("%d, %d, %d\n",idx, cell[idx],start[cell[idx]-x]);	}
	
	if (idx < nspts) {

		if (idx == 0 || cell[idx] != cell[idx - 1] ) {
			start[cell[idx]-x] = idx;
			start_copy[cell[idx]-x] = idx;
		}
		if (idx == nspts-1 || cell[idx] != cell[idx + 1] ) {
			end[cell[idx]-x] = idx;
		}
		//if (cell[idx]==16801){printf("fn: %d \n",idx);}
		//if (idx<10) {printf("%d, %d, %d, %d\n", x, idx, cell[idx]-x,start[cell[idx]-x]);}
	}
}

__global__ void mykernel(Particle *SPptr, int *particleindex, int *cell, int *start, int *end, int *split, int nspts,int x, int dev, int buffer, int *numsplit) {
	//x: start cellnumber
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	const int threadsperblockmax = 1024;

	int nb[27] = { -GRIDSIZE*GRIDSIZE - GRIDSIZE - 1, -GRIDSIZE*GRIDSIZE - GRIDSIZE,-GRIDSIZE*GRIDSIZE - GRIDSIZE + 1, -GRIDSIZE*GRIDSIZE - 1, -GRIDSIZE*GRIDSIZE, -GRIDSIZE*GRIDSIZE + 1, -GRIDSIZE*GRIDSIZE + GRIDSIZE - 1, -GRIDSIZE*GRIDSIZE + GRIDSIZE, -GRIDSIZE*GRIDSIZE + GRIDSIZE + 1,
		-GRIDSIZE - 1, -GRIDSIZE,-GRIDSIZE + 1, -1, 0, +1, GRIDSIZE - 1, GRIDSIZE, GRIDSIZE + 1,
		GRIDSIZE*GRIDSIZE - GRIDSIZE - 1, GRIDSIZE*GRIDSIZE - GRIDSIZE,GRIDSIZE*GRIDSIZE - GRIDSIZE + 1, GRIDSIZE*GRIDSIZE - 1, GRIDSIZE*GRIDSIZE, GRIDSIZE*GRIDSIZE + 1, GRIDSIZE*GRIDSIZE + GRIDSIZE - 1, GRIDSIZE*GRIDSIZE + GRIDSIZE, GRIDSIZE*GRIDSIZE + GRIDSIZE + 1 };

	
		//__shared__ int nb[27];
	//if (tidx < 27) {
	//				int x = demorton(bidx, 0);
	//				int y = demorton(bidx, 1);
	//				int z = demorton(bidx, 2);
	//				nb[tidx] = morton(x + tidx/9-1, y + (tidx/3)%3-1, z + tidx%3-1);
	//}
	//__syncthreads();
	

		//if (tidx == 0 && bidx< 24950 && start[bidx]>=0){printf("%d %d %d \n",bidx, start[bidx], SPptr[start[bidx]].cellnumber);}

	 __shared__ short int p[27];
	 	__shared__ short int pidx[27];
	 int __shared__ sum[threadsperblockmax];// = 0;
	 int __shared__ jj[threadsperblockmax];// = 0;
	volatile __shared__ int total;
	volatile 	__shared__ int blockpop;


	if (idx < threadsperblockmax)	{
		sum[idx] = 6500;
		jj[idx] = 6500;
	}
	__syncthreads();
	//__shared__ short int sum[27];
	//__shared__ short int j[27];
	//if (idx <nspts) { printf("%d, %d \n", idx, SPptr[idx].cellnumber); }
	if (tidx < 27) { p[tidx] = 0; }
	
	if (bidx < x && start[bidx] >= 0) {
		//if (bidx <50 ) { printf("%d %d %d\n", bidx, SPptr[particleindex[start[bidx]]].cellnumber, SPptr[particleindex[end[bidx]]].cellnumber); }
		///////////count and sort population of neighbour cells//////////////
		if (tidx < 27 && bidx+ nb[tidx] >= 0 && bidx + nb[tidx] < x+buffer && start[bidx + nb[tidx]] >= 0 &&end[bidx + nb[tidx]] >= 0 && start[bidx + nb[tidx]] < nspts && 1 + end[bidx + nb[tidx]] - start[bidx + nb[tidx]] > 0 ) {
			p[tidx] = 1 +end[bidx + nb[tidx]] - start[bidx + nb[tidx]]; //count population of neighbour cells so we know how many threads to use
			pidx[tidx] = tidx;
			
		}
		if (tidx == 13) { blockpop = p[tidx]; }
	}
	else {
		if (tidx == 13) { blockpop = 0; }
	}
	__syncthreads();

    if (blockpop > 6 && tidx < blockpop && bidx < x && start[bidx] >= 0 && particleindex[start[bidx]] >= 0){
        SPptr[particleindex[start[bidx]+tidx]].subindex = 1-(int((SPptr[particleindex[start[bidx]+tidx]].xcoord-XMIN)/CELLSIZE) == int((SPptr[particleindex[start[bidx]+tidx]].xcoord-XMIN+CELLSIZE/2)/CELLSIZE))
                                    + 2 - 2*(int((SPptr[particleindex[start[bidx]+tidx]].ycoord-YMIN)/CELLSIZE) == int((SPptr[particleindex[start[bidx]+tidx]].ycoord-YMIN+CELLSIZE/2)/CELLSIZE))
                                    +4*(int((SPptr[particleindex[start[bidx]+tidx]].zcoord-ZMIN)/CELLSIZE) == int((SPptr[particleindex[start[bidx]+tidx]].zcoord-ZMIN+CELLSIZE/2)/CELLSIZE));

		if (tidx==0){
			split[bidx] = bidx;
			atomicAdd(&(numsplit[0]), 1);
			//printf("blockpop = %d, bidx = %d, split[bidx] = %d\n",blockpop,bidx,split[bidx]);
		}
	}
	__syncthreads();
	

	//if (idx < x && start[idx] != -1) { printf("%d \n", start[idx]); }
	if (bidx < x && start[bidx] >= 0) {
		if (tidx == 0) {
			total = 0;
			for (int i = 0; i < 27; i++) {
				
				if (p[i]>0 && bidx + nb[i] >= 0 && bidx + nb[i] < x && start[bidx + nb[i]] >= 0 &&end[bidx + nb[i]] >= 0 && start[bidx + nb[i]] < nspts) { total += p[i];	}
			
			}
		}
	}
	else {
		if (tidx == 0) {total = 0; }
	}
	
	__syncthreads();
    if (total > threadsperblockmax && tidx ==0){printf("total = %d \n",total);}
    
	if (bidx<x && start[bidx] >= 0) {
		if (tidx == 0) {
			int count = 0;
			for (int i = 0; i < 27; i++) {
				if (p[i] != 0) {
					p[count++] = p[i];
					pidx[count - 1] = pidx[i]; //sort
				}
			}

			while (count < 27) {
				p[count++] = 0; //need to reset popidx in a future kernel
				pidx[count - 1] = 0;
			}
		}
	}
	__syncthreads();

	if (bidx<x && start[bidx] >= 0) {
		if (tidx < total) {
			sum[tidx] = 0;
			jj[tidx] = 0;
			while (tidx + 1 > sum[tidx] && jj[tidx] < threadsperblockmax) {
				sum[tidx] += p[jj[tidx]];
				jj[tidx]++;
			}
		}
	}
	__syncthreads();

	//if (bidx== 34624 && tidx < total) { printf("tidx: %d, cell#:%d, jj:%d, sum:%d \n", tidx, bidx + nb[pidx[jj[tidx] - 1]], jj[tidx],p[jj[tidx]]); }
	//__syncthreads();
	//int lb = 0;//dev*buffer; //lower bound cell
	//int hb = x+buffer; //higher bound cell
	//if (bidx ==0 && tidx ==0){printf("%d \n",*numsplit);}

	if (bidx<x && start[bidx] >= 0 && particleindex[start[bidx]] >= 0 && split[bidx] ==-1 && particleindex[start[bidx]] < nspts) { //should exclude the buffer here
		if (tidx < total && bidx + nb[pidx[jj[tidx] - 1]] >= 0 && bidx + nb[pidx[jj[tidx] - 1]] < x) { 
			///////////////////////////////////////////////////////////////////
			volatile int j = particleindex[start[bidx + nb[pidx[jj[tidx] - 1]]] + sum[tidx] - (tidx + 1)];


			if (particleindex[start[bidx + nb[pidx[jj[tidx] - 1]]]] >= 0 && j < nspts && j >= 0) {  
				for (volatile int i = start[bidx]; i <= end[bidx]; i++) {  
					volatile int ii = particleindex[i];
					float ds = (SPptr[ii]).distance((SPptr[j]));
					
					//Particle merging
					if (ds <= (-10.00) && ds > 0 && SPptr[ii].mass>0 && SPptr[j].mass> 0 && SPptr[ii].mass<2 && SPptr[j].mass<2 && !SPptr[ii].boundary &&!SPptr[j].boundary && powf(SPptr[ii].diffusionx,2)+powf(SPptr[ii].diffusiony,2)+powf(SPptr[ii].diffusionz,2) < 20 && powf(SPptr[j].diffusionx,2)+powf(SPptr[j].diffusiony,2)+powf(SPptr[j].diffusionz,2) < 20 ) {
						SPptr[ii].mass = 2.75;
						SPptr[j].mass = 0;
						
						SPptr[j].boundary = true;
						SPptr[ii].xvel=  (SPptr[ii].xvel + SPptr[j].xvel)/2.0;
						SPptr[ii].yvel=  (SPptr[ii].yvel + SPptr[j].yvel)/2.0;
						SPptr[ii].zvel=  (SPptr[ii].zvel + SPptr[j].zvel)/2.0;
						SPptr[ii].xcoord=  (SPptr[ii].xcoord + SPptr[j].xcoord)/2.0;
						SPptr[ii].ycoord=  (SPptr[ii].ycoord + SPptr[j].ycoord)/2.0;
						SPptr[ii].zcoord=  (SPptr[ii].zcoord + SPptr[j].zcoord)/2.0;
						SPptr[j].xcoord = SPptr[j].ycoord = SPptr[j].zcoord=90.99;
						//SPptr[j].cellnumber = NUMCELLS+1;
						//printf("%4.4f \n",ds);
					} 

					//Particle splitting
					if (SPptr[ii].mass>3 && SPptr[ii].cellnumber < NUMCELLS && !SPptr[ii].boundary && ((powf(SPptr[ii].diffusionx,2)+powf(SPptr[ii].diffusiony,2)+powf(SPptr[ii].diffusionz,2)) > 35000 || (SPptr[ii].dens < 9400))) {
						SPptr[ii].mass = 1;
						
						SPptr[ii].split = true;
						SPptr[ii].ycoord += 0.015;
						//SPptr[j].cellnumber = NUMCELLS+1;
						//printf("%4.4f \n",ds);
					}

					if (ds <= (2 * cutoff) && ds > 0) { 

						volatile float k = kernel(ds);
						volatile float rabx = (SPptr[ii]).rab_x((SPptr[j]));
						volatile float raby = (SPptr[ii]).rab_y((SPptr[j]));
						volatile float rabz = (SPptr[ii]).rab_z((SPptr[j]));
						volatile float vabx = (SPptr[ii]).vab_x((SPptr[j]));
						volatile float vaby = (SPptr[ii]).vab_y((SPptr[j]));
						volatile float vabz = (SPptr[ii]).vab_z((SPptr[j]));
						volatile float dkx = kernel_derivative(ds)*rabx / ds;
						volatile float dky = kernel_derivative(ds)*raby / ds;
						volatile float dkz = kernel_derivative(ds)*rabz / ds;

						//float dkxtest = kernel_test(ds)*rabx / ds;
						//float dkytest = kernel_test(ds)*raby / ds;
						//float dkztest = kernel_test(ds)*rabz / ds;

						volatile float d = dot_prod(vabx, vaby, vabz, rabx, raby, rabz);
						volatile float d2 = powf(ds, 2);
						//added mass
						volatile float s = (((SPptr[ii].solid*9+1)*ALPHA_FLUID)* SOUND * (powf(SPptr[i].mass,1)*cutoff * (d / (d2 + 0.01*powf(cutoff, 2))) + 50 * 1.0 / SOUND*powf(cutoff * (d / (d2 + 0.01*powf(cutoff, 2))), 2)) / (((SPptr[ii]).dens + (SPptr[j]).dens) / 2.0)) *(d < 0)*(1 + (!(SPptr[ii]).boundary)*((SPptr[j]).boundary) * ((1+3*SPptr[ii].fluid*SPptr[ii].fluid)*ALPHA__SAND_BOUNDARY));
						//float s2 = ALPHA_LAMINAR_FLUID * SOUND * cutoff / ((SPptr[i]).dens + (SPptr[j]).dens)*d*(d < 0) / (d2 + 0.01*pow(cutoff, 2))*(1 + (!(SPptr[i]).boundary)*((SPptr[j]).boundary) *ALPHA_LAMINAR_BOUNDARY); //laminar

						volatile float dpx = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[ii]).press / powf((SPptr[ii]).dens, 2) + s)*dkx;
						volatile float dpy = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[ii]).press / powf((SPptr[ii]).dens, 2) + s)*dky;
						volatile float dpz = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[ii]).press / powf((SPptr[ii]).dens, 2) + s)*dkz;

						volatile float mass_solid_frac = SPptr[ii].solid*RHO_0_SAND / (RHO_0_SAND*SPptr[ii].solid + RHO_0*(SPptr[ii].fluid));
						volatile float mass_fluid_frac = SPptr[ii].fluid*RHO_0 / (RHO_0_SAND*SPptr[ii].solid + RHO_0*(SPptr[ii].fluid));

						if (mass_solid_frac > 0.001 && mass_solid_frac < 0.999 && mass_fluid_frac > 0.001 && mass_fluid_frac < 0.999 && !SPptr[ii].boundary && !SPptr[j].boundary) {
							volatile float solidgradx = (SPptr[j].solid - SPptr[ii].solid)*dkx; //note that fluidgrad = -solidgrad if there are only 2 types of particles
							volatile float solidgrady = (SPptr[j].solid - SPptr[ii].solid)*dky;
							volatile float solidgradz = (SPptr[j].solid - SPptr[ii].solid)*dkz;

							float fluidgradx = (SPptr[j].fluid - SPptr[ii].fluid)*dkx;
							float fluidgrady = (SPptr[j].fluid - SPptr[ii].fluid)*dky;
							float fluidgradz = (SPptr[j].fluid - SPptr[ii].fluid)*dkz;

							float solidbrownianx = (solidgradx / (SPptr[ii].solid) - (mass_solid_frac*solidgradx / (SPptr[ii].solid) + mass_fluid_frac*fluidgradx / (SPptr[ii].fluid)));
							float solidbrowniany = (solidgrady / (SPptr[ii].solid) - (mass_solid_frac*solidgrady / (SPptr[ii].solid) + mass_fluid_frac*fluidgrady / (SPptr[ii].fluid)));
							float solidbrownianz = (solidgradz / (SPptr[ii].solid) - (mass_solid_frac*solidgradz / (SPptr[ii].solid) + mass_fluid_frac*fluidgradz / (SPptr[ii].fluid)));

							float fluidbrownianx = (fluidgradx / (SPptr[ii].fluid) - (mass_fluid_frac*fluidgradx / (SPptr[ii].fluid) + mass_solid_frac*solidgradx / (SPptr[ii].solid)));
							float fluidbrowniany = (fluidgrady / (SPptr[ii].fluid) - (mass_fluid_frac*fluidgrady / (SPptr[ii].fluid) + mass_solid_frac*solidgrady / (SPptr[ii].solid)));
							float fluidbrownianz = (fluidgradz / (SPptr[ii].fluid) - (mass_fluid_frac*fluidgradz / (SPptr[ii].fluid) + mass_solid_frac*solidgradz / (SPptr[ii].solid)));

							float solidpressureslipx = (SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dkx;
							float solidpressureslipy = (SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dky - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dky - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dky;
							float solidpressureslipz = (SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dkz;

							float fluidpressureslipx = (SPptr[ii].fluid*SPptr[ii].press - SPptr[j].fluid*SPptr[j].press)*dkx - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dkx;
							float fluidpressureslipy = (SPptr[ii].fluid*SPptr[ii].press - SPptr[j].fluid*SPptr[j].press)*dky - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dky - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dky;
							float fluidpressureslipz = (SPptr[ii].fluid*SPptr[ii].press - SPptr[j].fluid*SPptr[j].press)*dkz - mass_solid_frac*(SPptr[ii].solid*SPptr[ii].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_fluid_frac*((SPptr[ii].fluid)*SPptr[ii].press - (SPptr[j].fluid)*SPptr[j].press)*dkz;

							float solidbodyx = (SPptr[ii].solid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * ((150.0 / SPptr[ii].dens)*SPptr[ii].delpressx - SPptr[ii].xvel*dkx*vabx - SPptr[ii].yvel*dky*vabx - SPptr[ii].zvel*dkz*vabx);
							float solidbodyy = (SPptr[ii].solid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * ((150.0 / SPptr[ii].dens)*SPptr[ii].delpressy - SPptr[ii].xvel*dkx*vaby - SPptr[ii].yvel*dky*vaby - SPptr[ii].zvel*dkz*vaby);
							float solidbodyz = (SPptr[ii].solid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * (GRAVITY + (150.0 / SPptr[ii].dens)*SPptr[ii].delpressz - SPptr[ii].xvel*dkx*vabz - SPptr[ii].yvel*dky*vabz - SPptr[ii].zvel*dkz*vabz);

							float fluidbodyx = (SPptr[ii].fluid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * ((150.0 / SPptr[ii].dens)*SPptr[ii].delpressx - SPptr[ii].xvel*dkx*vabx - SPptr[ii].yvel*dky*vabx - SPptr[ii].zvel*dkz*vabx);
							float fluidbodyy = (SPptr[ii].fluid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * ((150.0 / SPptr[ii].dens)*SPptr[ii].delpressy - SPptr[ii].xvel*dkx*vaby - SPptr[ii].yvel*dky*vaby - SPptr[ii].zvel*dkz*vaby);
							float fluidbodyz = (SPptr[ii].fluid*SPptr[ii].dens - (mass_solid_frac*SPptr[ii].solid*SPptr[ii].dens + mass_fluid_frac*SPptr[ii].fluid*SPptr[ii].dens)) * (GRAVITY + (150.0 / SPptr[ii].dens)*SPptr[ii].delpressz - SPptr[ii].xvel*dkx*vabz - SPptr[ii].yvel*dky*vabz - SPptr[ii].zvel*dkz*vabz);
						
							atomicAdd(&(SPptr[ii].soliddriftvelx), MIXPRESSURE*(solidbodyx + solidpressureslipx) - MIXBROWNIAN*solidbrownianx);
							atomicAdd(&(SPptr[ii].soliddriftvely), MIXPRESSURE*(solidbodyy + solidpressureslipy) - MIXBROWNIAN*solidbrowniany);
							atomicAdd(&(SPptr[ii].soliddriftvelz), MIXPRESSURE*(solidbodyz + solidpressureslipz) - MIXBROWNIAN*solidbrownianz);

							atomicAdd(&(SPptr[ii].fluiddriftvelx), MIXPRESSURE*(fluidbodyx + fluidpressureslipx) - MIXBROWNIAN*fluidbrownianx);
							atomicAdd(&(SPptr[ii].fluiddriftvely), MIXPRESSURE*(fluidbodyy + fluidpressureslipy) - MIXBROWNIAN*fluidbrowniany);
							atomicAdd(&(SPptr[ii].fluiddriftvelz), MIXPRESSURE*(fluidbodyz + fluidpressureslipz) - MIXBROWNIAN*fluidbrownianz);
						}
							atomicAdd(&(SPptr[ii].newdelpressx), dpx*SPptr[j].mass); //added mass
							atomicAdd(&(SPptr[ii].newdelpressy), dpy*SPptr[j].mass);
							atomicAdd(&(SPptr[ii].newdelpressz), dpz*SPptr[j].mass);
							
							atomicAdd(&(SPptr[ii].newdens), k *(1 + float(!(SPptr[ii]).boundary)*float((SPptr[j]).boundary)*BDENSFACTOR)*SPptr[j].mass); //added mass

							atomicAdd(&(SPptr[ii].diffusionx),SPptr[j].mass/ SPptr[j].dens*dkx*!SPptr[j].boundary*!SPptr[ii].boundary); //added boundary
							atomicAdd(&(SPptr[ii].diffusiony), SPptr[j].mass/ SPptr[j].dens*dky*!SPptr[j].boundary*!SPptr[ii].boundary);
							atomicAdd(&(SPptr[ii].diffusionz), SPptr[j].mass / SPptr[j].dens*dkz*!SPptr[j].boundary*!SPptr[ii].boundary);
							
							volatile float mixfactor = (!SPptr[j].boundary)*(!SPptr[ii].boundary)*(SPptr[ii].solid > 0.0)* (SPptr[j].solid > 0.0) * 2 * (SPptr[ii].solid-0.0)*(SPptr[j].solid-0.0) / (SPptr[ii].solid-0.0 + SPptr[j].solid-0.0+0.01);
							atomicAdd(&(SPptr[ii].vel_grad[0][0]), -mixfactor*vabx*dkx *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[0][1]), -mixfactor*vaby*dkx *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[0][2]), -mixfactor*vabz*dkx *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[1][0]), -mixfactor*vabx*dky *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[1][1]), -mixfactor*vaby*dky *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[1][2]), -mixfactor*vabz*dky *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[2][0]), -mixfactor*vabx*dkz *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[2][1]), -mixfactor*vaby*dkz *1./ (SPptr[ii]).dens);
							atomicAdd(&(SPptr[ii].vel_grad[2][2]), -mixfactor*vabz*dkz *1./ (SPptr[ii]).dens);

							atomicAdd(&(SPptr[ii].stress_accel[0]), mixfactor*((SPptr[ii]).stress_tensor[0][0] * dkx + (SPptr[ii]).stress_tensor[0][1] * dky + (SPptr[ii]).stress_tensor[0][2] * dkz) / pow((SPptr[ii]).dens, 2) + ((SPptr[ii]).stress_tensor[0][0] * dkx + (SPptr[ii]).stress_tensor[0][1] * dky + (SPptr[ii]).stress_tensor[0][2] * dkz) / pow((SPptr[ii]).dens, 2));
							atomicAdd(&(SPptr[ii].stress_accel[1]), mixfactor*((SPptr[ii]).stress_tensor[1][0] * dkx + (SPptr[ii]).stress_tensor[1][1] * dky + (SPptr[ii]).stress_tensor[1][2] * dkz) / pow((SPptr[ii]).dens, 2) + ((SPptr[ii]).stress_tensor[1][0] * dkx + (SPptr[ii]).stress_tensor[1][1] * dky + (SPptr[ii]).stress_tensor[1][2] * dkz) / pow((SPptr[ii]).dens, 2));
							atomicAdd(&(SPptr[ii].stress_accel[2]), mixfactor*((SPptr[ii]).stress_tensor[2][0] * dkx + (SPptr[ii]).stress_tensor[2][1] * dky + (SPptr[ii]).stress_tensor[2][2] * dkz) / pow((SPptr[ii]).dens, 2) + ((SPptr[ii]).stress_tensor[2][0] * dkx + (SPptr[ii]).stress_tensor[2][1] * dky + (SPptr[ii]).stress_tensor[2][2] * dkz) / pow((SPptr[ii]).dens, 2));

							volatile float ds2 = dot_prod(SPptr[j].soliddriftvelx, SPptr[j].soliddriftvely, SPptr[j].soliddriftvelz, dkx, dky, dkz);
							volatile float ds = dot_prod(SPptr[ii].soliddriftvelx, SPptr[ii].soliddriftvely, SPptr[ii].soliddriftvelz, dkx, dky, dkz);

							volatile float df2 = dot_prod(SPptr[j].fluiddriftvelx, SPptr[j].fluiddriftvely, SPptr[j].fluiddriftvelz, dkx, dky, dkz);
							volatile float df = dot_prod(SPptr[ii].fluiddriftvelx, SPptr[ii].fluiddriftvely, SPptr[ii].fluiddriftvelz, dkx, dky, dkz);



							atomicAdd(&(SPptr[ii].mixture_accel[0]), -1 / SPptr[ii].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvelx*ds2 + SPptr[ii].solid*SPptr[ii].soliddriftvelx*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvelx*df2 + SPptr[ii].fluid*SPptr[ii].fluiddriftvelx*df)));

							atomicAdd(&(SPptr[ii].mixture_accel[1]), -1 / SPptr[ii].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvely*ds2 + SPptr[ii].solid*SPptr[ii].soliddriftvely*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvely*df2 + SPptr[ii].fluid*SPptr[ii].fluiddriftvely*df)));

							atomicAdd(&(SPptr[ii].mixture_accel[2]), -1 / SPptr[ii].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvelz*ds2 + SPptr[ii].solid*SPptr[ii].soliddriftvelz*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvelz*df2 + SPptr[ii].fluid*SPptr[ii].fluiddriftvelz*df)));

							atomicAdd(&(SPptr[ii].delsolid), (!SPptr[j].boundary) *(!SPptr[ii].boundary) *-0.5 / SPptr[j].dens*(SPptr[ii].solid + SPptr[j].solid) * (dkx*vabx + dky*vaby + dkz*vabz) + (-(SPptr[ii].solid*SPptr[ii].soliddriftvelx + SPptr[j].solid*SPptr[j].soliddriftvelx)*dkx - (SPptr[ii].solid*SPptr[ii].soliddriftvely + SPptr[j].solid*SPptr[j].soliddriftvely)*dky - (SPptr[ii].solid*SPptr[ii].soliddriftvelz + SPptr[j].solid*SPptr[j].soliddriftvelz)*dkz) / SPptr[j].dens);
							atomicAdd(&(SPptr[ii].delfluid), (!SPptr[j].boundary) *(!SPptr[ii].boundary) *-0.5 / SPptr[j].dens*(SPptr[ii].fluid + SPptr[j].fluid) * (dkx*vabx + dky*vaby + dkz*vabz) + (-(SPptr[ii].fluid*SPptr[ii].fluiddriftvelx + SPptr[j].fluid*SPptr[j].fluiddriftvelx)*dkx - (SPptr[ii].fluid*SPptr[ii].fluiddriftvely + SPptr[j].fluid*SPptr[j].fluiddriftvely)*dky - (SPptr[ii].fluid*SPptr[ii].fluiddriftvelz + SPptr[j].fluid*SPptr[j].fluiddriftvelz)*dkz) / SPptr[j].dens);
					}
				}
			}
		}
	}
	__syncthreads();


	if (idx < nspts) {
		if ((SPptr[idx]).solid) {
			float tr = 0; //trace of strain rate
			float tr2 = 0; //trace of stress tensor
			float tr3 = 0; //double dot of stress tensor
			float tr4 = 0; //trace of stress tensor times strain rate
			float tr5 = 0; //double dot of strain rate
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					(SPptr[idx]).strain_rate[p][q] = 0.5*((SPptr[idx]).vel_grad[p][q] + (SPptr[idx]).vel_grad[q][p]);
					
					tr3 += 0.5*(SPptr[idx]).stress_tensor[p][q]*(SPptr[idx]).stress_tensor[p][q];
					
					tr5 += (SPptr[idx]).strain_rate[p][q]*(SPptr[idx]).strain_rate[p][q];
					tr4 += (SPptr[idx]).stress_tensor[p][q] * (SPptr[idx]).strain_rate[q][p];
				}
				tr += (SPptr[idx]).strain_rate[p][p];
				tr2 += (SPptr[idx]).stress_tensor[p][p];



			}

			//	std::cout << (SPptr[index]).press << "\n";
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					if (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[idx]).press*(SPptr[idx].press>0) + KC / (sqrt(9 + 12 * pow(tan(PHI), 2))) < tr3 && tr3 != 0) {
						(SPptr[idx]).stress_tensor[p][q] *= (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[idx]).press*(SPptr[idx].press>0) + KC / (sqrt(9 + 12 * pow(tan(PHI), 2)))) / tr3;
					}
					(SPptr[idx]).stress_rate[p][q] = 3 * C1*((SPptr[idx]).press)*((SPptr[idx]).strain_rate[p][q] - 1. / 3.*tr*(p == q)) + C1*C2*(tr4 + tr*(SPptr[idx]).press*(SPptr[idx].press>0)) / (pow((SPptr[idx]).press, 2) + 1e8)*(SPptr[idx]).stress_tensor[p][q] - C1*C3*sqrt(tr5)*(SPptr[idx]).stress_tensor[p][q];
					//std::cout << tr4 << ", " << tr*(SPptr[index]).press << "\n";

				}
			}
		}
		//}	
	}
		__syncthreads();

	}

__global__ void mykernel2(Particle *SPptr, int *particleindex, int *cells, int *start_copy, int *start, int *end, int *split, int *numsplit, int nspts, int x, int dev, int buffer, int t, float *spts, float *a3, float *b3) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int lb = max(0, dev*NUMCELLS/2-buffer);
	int hb = lb+x;//min(NUMCELLS, NUMCELLS/4*(dev+1)+buffer);
	
	 
	if (index < nspts) {
        SPptr[particleindex[index]].flag = false;
		if (!(SPptr[particleindex[index]]).flag&&SPptr[particleindex[index]].cellnumber >= lb && SPptr[particleindex[index]].cellnumber < hb) {
			//printf("%d %d %d\n",dev, index, SPptr[index].cellnumber);
			spts[(3 * index)] = (SPptr[particleindex[index]]).xcoord;
			spts[(3 * index) + 1] = (SPptr[particleindex[index]]).ycoord;
			spts[(3 * index) + 2] = (SPptr[particleindex[index]]).zcoord;
			a3[index] = ((SPptr[particleindex[index]])).mass;
			b3[index] = powf(SPptr[particleindex[index]].diffusionx,2)+powf(SPptr[particleindex[index]].diffusiony,2)+powf(SPptr[particleindex[index]].diffusionz,2);
		}
		//if (SPptr[index].cellnumber >= dev*NUMCELLS/2 && SPptr[index].cellnumber < dev*NUMCELLS/2 + NUMCELLS/2){ //for 2 devices
		(SPptr[particleindex[index]]).update(t);
		//}
		
		//(SPptr[index]).cellnumber = int((SPptr[index].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[index].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[index].zcoord - ZMIN) / CELLSIZE);
		
		cells[index] = SPptr[particleindex[index]].cellnumber; 
		SPptr[index].newdens = 0;
		SPptr[index].newdelpressx = 0;
		SPptr[index].newdelpressy = 0;
		SPptr[index].newdelpressz = 0;
		SPptr[index].vel_grad[0][0] = SPptr[index].vel_grad[0][1] = SPptr[index].vel_grad[0][2] = SPptr[index].vel_grad[1][0] = SPptr[index].vel_grad[1][1] = SPptr[index].vel_grad[1][2] = SPptr[index].vel_grad[2][0] = SPptr[index].vel_grad[2][1] = SPptr[index].vel_grad[2][2] = 0 ;
		SPptr[index].stress_accel[0] = SPptr[index].stress_accel[1] = SPptr[index].stress_accel[2] = 0;
		SPptr[index].fluiddriftvelx	= SPptr[index].fluiddriftvely = SPptr[index].fluiddriftvelz = SPptr[index].soliddriftvelx = SPptr[index].soliddriftvely = SPptr[index].soliddriftvelz = 0;
		SPptr[index].mixture_accel[0] = SPptr[index].mixture_accel[1] = SPptr[index].mixture_accel[2] = 0;
		SPptr[index].delsolid = SPptr[index].delfluid = SPptr[index].diffusionx = SPptr[index].diffusionz =SPptr[index].diffusiony =0;
		
	}
	if (index < x) {
		start[index] = -1;
		start_copy[index] = -1;
		end[index] = -1;
		split[index] = -1;
	}

	if (index ==0) {numsplit[0]=0;}

	
	__syncthreads();
}

__global__ void find_idx(int *SPptr, int dev, int npts, int buffer, int *xleft, int *xright, int *sleft, int *sright) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lb = dev*NUMCELLS/2;
    int hb = lb + NUMCELLS/2;
    	
    if (idx < npts){
        if (dev == 0 && SPptr[idx+1]>= hb && SPptr[idx]<hb){
            xleft[0] = 0; // index of first updated particle
            xright[0] = idx; // index of final updated particle
            sleft[0] = 0; //index of final particle to transfer to dev-1
           // printf("xright: %d %d %d \n",idx,  SPptr[idx+1], SPptr[idx]);
		}
		if (dev == 0 && SPptr[idx+1]>= hb-buffer && SPptr[idx]<hb-buffer){
			sright[0] = idx+1;
			//printf("sright: %d %d %d \n",idx+1,  SPptr[idx], SPptr[idx+1]);
		}
		
        if (dev == 1 && SPptr[idx]<lb && SPptr[idx+1]>=lb){
            xleft[0] = idx+1;
			xright[0] = npts-1;
			sright[0] = npts; //index of first particle to transfer to dev+1
			//printf("xleft: %d %d %d \n",idx+1,  SPptr[idx], SPptr[idx+1]);
			//printf("xright: %d %d %d \n",npts-1,  SPptr[npts], SPptr[npts-1]);
		}
		if (dev == 1 && SPptr[idx]<lb+buffer && SPptr[idx+1]>=lb+buffer){
			sleft[0] = idx+1;
			//printf("sleft: %d %d %d \n",idx+1,  SPptr[idx], SPptr[idx+1]);
        }
    }
  //  __syncthreads();
}

__global__ void mem_shift(Particle *SPptr, Particle *buff, int *cells, int *ibuff, int dev, int shifts, int indexleft, int indexright) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (shifts !=0 && idx <= indexright && idx >= indexleft){
		buff[idx-indexleft] = SPptr[idx];
	}
	
	__syncthreads();
	if (shifts !=0 && idx <= indexright && idx >= indexleft){
		SPptr[idx-shifts] = buff[idx-indexleft];
	}
    
}

__global__ void cell_calc(Particle *SPptr, int *particleindex, int *cells, int size, int dev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx< size){
		(SPptr[particleindex[idx]]).cellnumber = int((SPptr[particleindex[idx]].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[particleindex[idx]].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[particleindex[idx]].zcoord - ZMIN) / CELLSIZE);
		cells[idx] = SPptr[particleindex[idx]].cellnumber;
	}

}


__global__ void count_after_merge(int *cells, int *particleindex, int size, int *newsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > 0 && idx < size){
		if (cells[idx]>=NUMCELLS & cells[idx-1]<NUMCELLS){
			*newsize = idx;
		}
	}

}






__global__ void mykernel3(Particle *SPptr, int *particleindex, int *cell, int *start, int *end, int *split, int nspts,int x, int dev, int buffer, int *numsplit) {

	//x: start cellnumber
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	const int threadsperblockmax = 1024;


	//bidx checks cell number split[bidx/8], subindex bidx % 8
	int dirx = (bidx % 8) & 1;
	int diry = ((bidx % 8) & 2) >> 1;
	int dirz = ((bidx % 8) & 4) >> 2;

	int nb[8] = {0, pow(-1,1+dirx)*GRIDSIZE*GRIDSIZE, pow(-1,1+diry)*GRIDSIZE, pow(-1,dirz),pow(-1,1+dirx)*GRIDSIZE*GRIDSIZE + pow(-1,1+diry)*GRIDSIZE, pow(-1,1+dirx)*GRIDSIZE*GRIDSIZE + pow(-1,dirz), pow(-1,1+diry)*GRIDSIZE + pow(-1,dirz), pow(-1,1+dirx)*GRIDSIZE*GRIDSIZE + pow(-1,1+diry)*GRIDSIZE + pow(-1,dirz)};
	//if (split[bidx/8] >=0 && bidx % 8 == 0 && tidx < 8 ){printf("%d %d\n",split[bidx/8], split[bidx/8] + nb[tidx]);}

	__shared__ short int p[8];
	__shared__ short int pidx[8];
	 int __shared__ sum[threadsperblockmax];// = 0;
	 int __shared__ jj[threadsperblockmax];// = 0;
	volatile __shared__ int total;
	volatile 	__shared__ int blockpop;


	if (idx < threadsperblockmax)	{
		sum[idx] = 6500;
		jj[idx] = 6500;
	}
	__syncthreads();
	//__shared__ short int sum[27];
	//__shared__ short int j[27];
	//if (idx <nspts) { printf("%d, %d \n", idx, SPptr[idx].cellnumber); }
	if (tidx < 8) { p[tidx] = 0; }
	

	//REPLACE [bidx] with [split[bidx/8]] and limit tidx to 8. tidx == 13 to tidx == 0
	if (split[bidx/8] < x && start[split[bidx/8]] >= 0) {
		///////////count and sort population of neighbour cells//////////////
		if (tidx < 8 && split[bidx/8] + nb[tidx] >= 0 && split[bidx/8] + nb[tidx] < x+buffer && start[split[bidx/8] + nb[tidx]] >= 0 && end[split[bidx/8] + nb[tidx]] >= 0 && start[split[bidx/8] + nb[tidx]] < nspts && 1 + end[split[bidx/8] + nb[tidx]] - start[split[bidx/8] + nb[tidx]] > 0 ) {
			p[tidx] = 1 + end[split[bidx/8] + nb[tidx]] - start[split[bidx/8] + nb[tidx]]; //count population of neighbour cells so we know how many threads to use
			pidx[tidx] = tidx;
			
		}
		if (tidx == 0) { blockpop = p[tidx]; }
	}
	else {
		if (tidx == 0) { blockpop = 0; }
	}
	__syncthreads();
	
	
	if (split[bidx/8] < x && start[split[bidx/8]] >= 0) {
		if (tidx == 0) {
			total = 0;
			for (int i = 0; i < 8; i++) {
				
				if (p[i]>0 && split[bidx/8] + nb[i] >= 0 && split[bidx/8] + nb[i] < x && start[split[bidx/8] + nb[i]] >= 0 && end[split[bidx/8] + nb[i]] >= 0 && start[split[bidx/8] + nb[i]] < nspts) { total += p[i];	}
			
			}
		}
	}
	else {
		if (tidx == 0) {total = 0; }
	}
	
	__syncthreads();
    if (total > threadsperblockmax && tidx ==0){printf("total = %d \n",total);}
    
	if (split[bidx/8]<x &&start[split[bidx/8]] >= 0) {
		if (tidx == 0) {
			int count = 0;
			for (int i = 0; i < 8; i++) {
				if (p[i] != 0) {
					p[count++] = p[i];
					pidx[count - 1] = pidx[i]; //sort
				}
			}

			while (count < 8) {
				p[count++] = 0; //need to reset popidx in a future kernel
				pidx[count - 1] = 0;
			}
		}
	}
	__syncthreads();

	if (split[bidx/8]<x && start[split[bidx/8]] >= 0) {
		if (tidx < total) {
			sum[tidx] = 0;
			jj[tidx] = 0;
			while (tidx + 1 > sum[tidx] && jj[tidx] < threadsperblockmax) { //used to be 64? i think threadsperblockmax makes more sense
				sum[tidx] += p[jj[tidx]];
				jj[tidx]++;
			}
		}
	}
	__syncthreads();

	if (split[bidx/8]<x && start[split[bidx/8]] >= 0 && split[bidx/8] >=0 && start[split[bidx/8]] < nspts) { //should exclude the buffer here
		if (tidx < total && split[bidx/8] + nb[pidx[jj[tidx] - 1]] >= 0 && split[bidx/8] + nb[pidx[jj[tidx] - 1]] < x) { 
			///////////////////////////////////////////////////////////////////
			volatile int j = particleindex[start[split[bidx/8] + nb[pidx[jj[tidx] - 1]]] + sum[tidx] - (tidx + 1)];


			if (start[split[bidx/8] + nb[pidx[jj[tidx] - 1]]] >= 0 && j < nspts && j >= 0) {  
				for (volatile int ii = start[split[bidx/8]]; ii <= end[split[bidx/8]]; ii++) {  //ADD ANOTHER CHECK HERE TO MATCH SUBINDEX TO BIDX % 8
					volatile int i = particleindex[ii];
					if (SPptr[i].subindex == bidx % 8){
					float ds = (SPptr[i]).distance((SPptr[j]));
					
					//Particle merging
					if (ds <= (-10.00) && ds > 0 && SPptr[i].mass>0 && SPptr[j].mass> 0 && SPptr[i].mass<2 && SPptr[j].mass<2 && !SPptr[i].boundary &&!SPptr[j].boundary && powf(SPptr[i].diffusionx,2)+powf(SPptr[i].diffusiony,2)+powf(SPptr[i].diffusionz,2) < 20 && powf(SPptr[j].diffusionx,2)+powf(SPptr[j].diffusiony,2)+powf(SPptr[j].diffusionz,2) < 20) {
						SPptr[ii].mass = 2.75;
						SPptr[j].mass = 0;
						
						SPptr[j].boundary = true;
						SPptr[ii].xvel=  (SPptr[ii].xvel + SPptr[j].xvel)/2.0;
						SPptr[ii].yvel=  (SPptr[ii].yvel + SPptr[j].yvel)/2.0;
						SPptr[ii].zvel=  (SPptr[ii].zvel + SPptr[j].zvel)/2.0;
						SPptr[ii].xcoord=  (SPptr[ii].xcoord + SPptr[j].xcoord)/2.0;
						SPptr[ii].ycoord=  (SPptr[ii].ycoord + SPptr[j].ycoord)/2.0;
						SPptr[ii].zcoord=  (SPptr[ii].zcoord + SPptr[j].zcoord)/2.0;
						SPptr[j].xcoord = SPptr[j].ycoord = SPptr[j].zcoord=90.99;
						//SPptr[j].cellnumber = NUMCELLS+1;
						//printf("%4.4f \n",ds);
					} 

					//Particle splitting
					if (SPptr[i].mass>3 && SPptr[i].cellnumber < NUMCELLS && !SPptr[i].boundary && ((powf(SPptr[i].diffusionx,2)+powf(SPptr[i].diffusiony,2)+powf(SPptr[i].diffusionz,2)) > 35000 || (SPptr[i].dens < 9400))) {
						SPptr[i].mass = 1;
						
						SPptr[i].split = true;
						SPptr[i].ycoord += 0.015;
						//SPptr[j].cellnumber = NUMCELLS+1;
						//printf("%4.4f \n",ds);
					}

					if (ds <= (2 * cutoff) && ds > 0) { //should exclude buffer here somehow?

						volatile float k = kernel(ds);
						volatile float rabx = (SPptr[i]).rab_x((SPptr[j]));
						volatile float raby = (SPptr[i]).rab_y((SPptr[j]));
						volatile float rabz = (SPptr[i]).rab_z((SPptr[j]));
						volatile float vabx = (SPptr[i]).vab_x((SPptr[j]));
						volatile float vaby = (SPptr[i]).vab_y((SPptr[j]));
						volatile float vabz = (SPptr[i]).vab_z((SPptr[j]));
						volatile float dkx = kernel_derivative(ds)*rabx / ds;
						volatile float dky = kernel_derivative(ds)*raby / ds;
						volatile float dkz = kernel_derivative(ds)*rabz / ds;

						//float dkxtest = kernel_test(ds)*rabx / ds;
						//float dkytest = kernel_test(ds)*raby / ds;
						//float dkztest = kernel_test(ds)*rabz / ds;

						volatile float d = dot_prod(vabx, vaby, vabz, rabx, raby, rabz);
						volatile float d2 = powf(ds, 2);
						//added mass
						volatile float s = (((SPptr[i].solid*9+1)*ALPHA_FLUID)* SOUND * (powf(SPptr[i].mass,1)*cutoff * (d / (d2 + 0.01*powf(cutoff, 2))) + 50 * 1.0 / SOUND*powf(cutoff * (d / (d2 + 0.01*powf(cutoff, 2))), 2)) / (((SPptr[i]).dens + (SPptr[j]).dens) / 2.0)) *(d < 0)*(1 + (!(SPptr[i]).boundary)*((SPptr[j]).boundary) * ((1+3*SPptr[i].fluid*SPptr[i].fluid)*ALPHA__SAND_BOUNDARY));
						//float s2 = ALPHA_LAMINAR_FLUID * SOUND * cutoff / ((SPptr[i]).dens + (SPptr[j]).dens)*d*(d < 0) / (d2 + 0.01*pow(cutoff, 2))*(1 + (!(SPptr[i]).boundary)*((SPptr[j]).boundary) *ALPHA_LAMINAR_BOUNDARY); //laminar

						volatile float dpx = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dkx;
						volatile float dpy = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dky;
						volatile float dpz = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dkz;

						volatile float mass_solid_frac = SPptr[i].solid*RHO_0_SAND / (RHO_0_SAND*SPptr[i].solid + RHO_0*(SPptr[i].fluid));
						volatile float mass_fluid_frac = SPptr[i].fluid*RHO_0 / (RHO_0_SAND*SPptr[i].solid + RHO_0*(SPptr[i].fluid));

						if (mass_solid_frac > 0.001 && mass_solid_frac < 0.999 && mass_fluid_frac > 0.001 && mass_fluid_frac < 0.999 && !SPptr[i].boundary && !SPptr[j].boundary) {
							volatile float solidgradx = (SPptr[j].solid - SPptr[i].solid)*dkx; //note that fluidgrad = -solidgrad if there are only 2 types of particles
							volatile float solidgrady = (SPptr[j].solid - SPptr[i].solid)*dky;
							volatile float solidgradz = (SPptr[j].solid - SPptr[i].solid)*dkz;

							float fluidgradx = (SPptr[j].fluid - SPptr[i].fluid)*dkx;
							float fluidgrady = (SPptr[j].fluid - SPptr[i].fluid)*dky;
							float fluidgradz = (SPptr[j].fluid - SPptr[i].fluid)*dkz;

							float solidbrownianx = (solidgradx / (SPptr[i].solid) - (mass_solid_frac*solidgradx / (SPptr[i].solid) + mass_fluid_frac*fluidgradx / (SPptr[i].fluid)));
							float solidbrowniany = (solidgrady / (SPptr[i].solid) - (mass_solid_frac*solidgrady / (SPptr[i].solid) + mass_fluid_frac*fluidgrady / (SPptr[i].fluid)));
							float solidbrownianz = (solidgradz / (SPptr[i].solid) - (mass_solid_frac*solidgradz / (SPptr[i].solid) + mass_fluid_frac*fluidgradz / (SPptr[i].fluid)));

							float fluidbrownianx = (fluidgradx / (SPptr[i].fluid) - (mass_fluid_frac*fluidgradx / (SPptr[i].fluid) + mass_solid_frac*solidgradx / (SPptr[i].solid)));
							float fluidbrowniany = (fluidgrady / (SPptr[i].fluid) - (mass_fluid_frac*fluidgrady / (SPptr[i].fluid) + mass_solid_frac*solidgrady / (SPptr[i].solid)));
							float fluidbrownianz = (fluidgradz / (SPptr[i].fluid) - (mass_fluid_frac*fluidgradz / (SPptr[i].fluid) + mass_solid_frac*solidgradz / (SPptr[i].solid)));

							float solidpressureslipx = (SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dkx;
							float solidpressureslipy = (SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dky - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dky - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dky;
							float solidpressureslipz = (SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dkz;

							float fluidpressureslipx = (SPptr[i].fluid*SPptr[i].press - SPptr[j].fluid*SPptr[j].press)*dkx - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkx - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dkx;
							float fluidpressureslipy = (SPptr[i].fluid*SPptr[i].press - SPptr[j].fluid*SPptr[j].press)*dky - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dky - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dky;
							float fluidpressureslipz = (SPptr[i].fluid*SPptr[i].press - SPptr[j].fluid*SPptr[j].press)*dkz - mass_solid_frac*(SPptr[i].solid*SPptr[i].press - SPptr[j].solid*SPptr[j].press)*dkz - mass_fluid_frac*((SPptr[i].fluid)*SPptr[i].press - (SPptr[j].fluid)*SPptr[j].press)*dkz;

							float solidbodyx = (SPptr[i].solid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * ((150.0 / SPptr[i].dens)*SPptr[i].delpressx - SPptr[i].xvel*dkx*vabx - SPptr[i].yvel*dky*vabx - SPptr[i].zvel*dkz*vabx);
							float solidbodyy = (SPptr[i].solid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * ((150.0 / SPptr[i].dens)*SPptr[i].delpressy - SPptr[i].xvel*dkx*vaby - SPptr[i].yvel*dky*vaby - SPptr[i].zvel*dkz*vaby);
							float solidbodyz = (SPptr[i].solid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * (GRAVITY + (150.0 / SPptr[i].dens)*SPptr[i].delpressz - SPptr[i].xvel*dkx*vabz - SPptr[i].yvel*dky*vabz - SPptr[i].zvel*dkz*vabz);

							float fluidbodyx = (SPptr[i].fluid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * ((150.0 / SPptr[i].dens)*SPptr[i].delpressx - SPptr[i].xvel*dkx*vabx - SPptr[i].yvel*dky*vabx - SPptr[i].zvel*dkz*vabx);
							float fluidbodyy = (SPptr[i].fluid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * ((150.0 / SPptr[i].dens)*SPptr[i].delpressy - SPptr[i].xvel*dkx*vaby - SPptr[i].yvel*dky*vaby - SPptr[i].zvel*dkz*vaby);
							float fluidbodyz = (SPptr[i].fluid*SPptr[i].dens - (mass_solid_frac*SPptr[i].solid*SPptr[i].dens + mass_fluid_frac*SPptr[i].fluid*SPptr[i].dens)) * (GRAVITY + (150.0 / SPptr[i].dens)*SPptr[i].delpressz - SPptr[i].xvel*dkx*vabz - SPptr[i].yvel*dky*vabz - SPptr[i].zvel*dkz*vabz);
						
							atomicAdd(&(SPptr[i].soliddriftvelx), MIXPRESSURE*(solidbodyx + solidpressureslipx) - MIXBROWNIAN*solidbrownianx);
							atomicAdd(&(SPptr[i].soliddriftvely), MIXPRESSURE*(solidbodyy + solidpressureslipy) - MIXBROWNIAN*solidbrowniany);
							atomicAdd(&(SPptr[i].soliddriftvelz), MIXPRESSURE*(solidbodyz + solidpressureslipz) - MIXBROWNIAN*solidbrownianz);

							atomicAdd(&(SPptr[i].fluiddriftvelx), MIXPRESSURE*(fluidbodyx + fluidpressureslipx) - MIXBROWNIAN*fluidbrownianx);
							atomicAdd(&(SPptr[i].fluiddriftvely), MIXPRESSURE*(fluidbodyy + fluidpressureslipy) - MIXBROWNIAN*fluidbrowniany);
							atomicAdd(&(SPptr[i].fluiddriftvelz), MIXPRESSURE*(fluidbodyz + fluidpressureslipz) - MIXBROWNIAN*fluidbrownianz);
						}
							atomicAdd(&(SPptr[i].newdelpressx), dpx*SPptr[j].mass); //added mass
							atomicAdd(&(SPptr[i].newdelpressy), dpy*SPptr[j].mass);
							atomicAdd(&(SPptr[i].newdelpressz), dpz*SPptr[j].mass);
							
							atomicAdd(&(SPptr[i].newdens), k *(1 + float(!(SPptr[i]).boundary)*float((SPptr[j]).boundary)*BDENSFACTOR)*SPptr[j].mass); //added mass

							atomicAdd(&(SPptr[i].diffusionx), SPptr[j].mass / SPptr[j].dens*dkx*!SPptr[j].boundary*!SPptr[i].boundary); //added mass
							atomicAdd(&(SPptr[i].diffusiony), SPptr[j].mass / SPptr[j].dens*dky*!SPptr[j].boundary*!SPptr[i].boundary);
							atomicAdd(&(SPptr[i].diffusionz), SPptr[j].mass / SPptr[j].dens*dkz*!SPptr[j].boundary*!SPptr[i].boundary);
							
							volatile float mixfactor = (!SPptr[j].boundary)*(!SPptr[i].boundary)*(SPptr[i].solid > 0.0)* (SPptr[j].solid > 0.0) * 2 * (SPptr[i].solid-0.0)*(SPptr[j].solid-0.0) / (SPptr[i].solid-0.0 + SPptr[j].solid-0.0+0.01);
							atomicAdd(&(SPptr[i].vel_grad[0][0]), -mixfactor*vabx*dkx *1./ (SPptr[i]).dens); //added mass
							atomicAdd(&(SPptr[i].vel_grad[0][1]), -mixfactor*vaby*dkx *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[0][2]), -mixfactor*vabz*dkx *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[1][0]), -mixfactor*vabx*dky *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[1][1]), -mixfactor*vaby*dky *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[1][2]), -mixfactor*vabz*dky *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[2][0]), -mixfactor*vabx*dkz *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[2][1]), -mixfactor*vaby*dkz *1./ (SPptr[i]).dens);
							atomicAdd(&(SPptr[i].vel_grad[2][2]), -mixfactor*vabz*dkz *1./ (SPptr[i]).dens);

							atomicAdd(&(SPptr[i].stress_accel[0]), mixfactor*((SPptr[i]).stress_tensor[0][0] * dkx + (SPptr[i]).stress_tensor[0][1] * dky + (SPptr[i]).stress_tensor[0][2] * dkz) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[0][0] * dkx + (SPptr[i]).stress_tensor[0][1] * dky + (SPptr[i]).stress_tensor[0][2] * dkz) / pow((SPptr[i]).dens, 2));
							atomicAdd(&(SPptr[i].stress_accel[1]), mixfactor*((SPptr[i]).stress_tensor[1][0] * dkx + (SPptr[i]).stress_tensor[1][1] * dky + (SPptr[i]).stress_tensor[1][2] * dkz) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[1][0] * dkx + (SPptr[i]).stress_tensor[1][1] * dky + (SPptr[i]).stress_tensor[1][2] * dkz) / pow((SPptr[i]).dens, 2));
							atomicAdd(&(SPptr[i].stress_accel[2]), mixfactor*((SPptr[i]).stress_tensor[2][0] * dkx + (SPptr[i]).stress_tensor[2][1] * dky + (SPptr[i]).stress_tensor[2][2] * dkz) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[2][0] * dkx + (SPptr[i]).stress_tensor[2][1] * dky + (SPptr[i]).stress_tensor[2][2] * dkz) / pow((SPptr[i]).dens, 2));

							volatile float ds2 = dot_prod(SPptr[j].soliddriftvelx, SPptr[j].soliddriftvely, SPptr[j].soliddriftvelz, dkx, dky, dkz);
							volatile float ds = dot_prod(SPptr[i].soliddriftvelx, SPptr[i].soliddriftvely, SPptr[i].soliddriftvelz, dkx, dky, dkz);

							volatile float df2 = dot_prod(SPptr[j].fluiddriftvelx, SPptr[j].fluiddriftvely, SPptr[j].fluiddriftvelz, dkx, dky, dkz);
							volatile float df = dot_prod(SPptr[i].fluiddriftvelx, SPptr[i].fluiddriftvely, SPptr[i].fluiddriftvelz, dkx, dky, dkz);



							atomicAdd(&(SPptr[i].mixture_accel[0]), -1 / SPptr[i].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvelx*ds2 + SPptr[i].solid*SPptr[i].soliddriftvelx*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvelx*df2 + SPptr[i].fluid*SPptr[i].fluiddriftvelx*df)));

							atomicAdd(&(SPptr[i].mixture_accel[1]), -1 / SPptr[i].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvely*ds2 + SPptr[i].solid*SPptr[i].soliddriftvely*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvely*df2 + SPptr[i].fluid*SPptr[i].fluiddriftvely*df)));

							atomicAdd(&(SPptr[i].mixture_accel[2]), -1 / SPptr[i].dens / SPptr[j].dens*(SPptr[j].solid*SPptr[j].dens*(SPptr[j].solid*SPptr[j].soliddriftvelz*ds2 + SPptr[i].solid*SPptr[i].soliddriftvelz*ds)
								+ SPptr[j].fluid*SPptr[j].dens*(SPptr[j].fluid*SPptr[j].fluiddriftvelz*df2 + SPptr[i].fluid*SPptr[i].fluiddriftvelz*df)));

							atomicAdd(&(SPptr[i].delsolid), (!SPptr[j].boundary) *(!SPptr[i].boundary) *-0.5 / SPptr[j].dens*(SPptr[i].solid + SPptr[j].solid) * (dkx*vabx + dky*vaby + dkz*vabz) + (-(SPptr[i].solid*SPptr[i].soliddriftvelx + SPptr[j].solid*SPptr[j].soliddriftvelx)*dkx - (SPptr[i].solid*SPptr[i].soliddriftvely + SPptr[j].solid*SPptr[j].soliddriftvely)*dky - (SPptr[i].solid*SPptr[i].soliddriftvelz + SPptr[j].solid*SPptr[j].soliddriftvelz)*dkz) / SPptr[j].dens);
							atomicAdd(&(SPptr[i].delfluid), (!SPptr[j].boundary) *(!SPptr[i].boundary) *-0.5 / SPptr[j].dens*(SPptr[i].fluid + SPptr[j].fluid) * (dkx*vabx + dky*vaby + dkz*vabz) + (-(SPptr[i].fluid*SPptr[i].fluiddriftvelx + SPptr[j].fluid*SPptr[j].fluiddriftvelx)*dkx - (SPptr[i].fluid*SPptr[i].fluiddriftvely + SPptr[j].fluid*SPptr[j].fluiddriftvely)*dky - (SPptr[i].fluid*SPptr[i].fluiddriftvelz + SPptr[j].fluid*SPptr[j].fluiddriftvelz)*dkz) / SPptr[j].dens);
					}
				}
				}
			}
		}
	}
	__syncthreads();


	if (idx < nspts) {
		if ((SPptr[idx]).solid) {
			float tr = 0; //trace of strain rate
			float tr2 = 0; //trace of stress tensor
			float tr3 = 0; //double dot of stress tensor
			float tr4 = 0; //trace of stress tensor times strain rate
			float tr5 = 0; //double dot of strain rate
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					(SPptr[idx]).strain_rate[p][q] = 0.5*((SPptr[idx]).vel_grad[p][q] + (SPptr[idx]).vel_grad[q][p]);
					
					tr3 += 0.5*(SPptr[idx]).stress_tensor[p][q]*(SPptr[idx]).stress_tensor[p][q];
					
					tr5 += (SPptr[idx]).strain_rate[p][q]*(SPptr[idx]).strain_rate[p][q];
					tr4 += (SPptr[idx]).stress_tensor[p][q] * (SPptr[idx]).strain_rate[q][p];
				}
				tr += (SPptr[idx]).strain_rate[p][p];
				tr2 += (SPptr[idx]).stress_tensor[p][p];



			}

			//	std::cout << (SPptr[index]).press << "\n";
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					if (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[idx]).press*(SPptr[idx].press>0) + KC / (sqrt(9 + 12 * pow(tan(PHI), 2))) < tr3 && tr3 != 0) {
						(SPptr[idx]).stress_tensor[p][q] *= (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[idx]).press*(SPptr[idx].press>0) + KC / (sqrt(9 + 12 * pow(tan(PHI), 2)))) / tr3;
					}
					(SPptr[idx]).stress_rate[p][q] = 3 * C1*((SPptr[idx]).press)*((SPptr[idx]).strain_rate[p][q] - 1. / 3.*tr*(p == q)) + C1*C2*(tr4 + tr*(SPptr[idx]).press*(SPptr[idx].press>0)) / (pow((SPptr[idx]).press, 2) + 1e8)*(SPptr[idx]).stress_tensor[p][q] - C1*C3*sqrt(tr5)*(SPptr[idx]).stress_tensor[p][q];
					//std::cout << tr4 << ", " << tr*(SPptr[index]).press << "\n";

				}
			}
		}
		//}	
	}
		__syncthreads();

	}