#include "FluidGPU.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
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



__global__ void findneighbours(int *cell, int *start, int *end, int nspts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nspts) {
		if (cell[idx] != cell[idx - 1] || idx == 0) {
			start[cell[idx]] = idx;
		}
		if (cell[idx] != cell[idx + 1] || idx == nspts-1) {
			end[cell[idx]] = idx;
		}

	}
}

__global__ void mykernel(Particle *SPptr, int *cell, int *start, int *end, int nspts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;

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
	
	 __shared__ short int p[27];
	 	__shared__ short int pidx[27];
	 int __shared__ sum[64];// = 0;
	 int __shared__ jj[64];// = 0;
	volatile __shared__ int total;
	volatile 	__shared__ int blockpop;
	if (idx < 64)	{
		sum[idx] = 650;
		jj[idx] = 650;
	}
	__syncthreads();
	//__shared__ short int sum[27];
	//__shared__ short int j[27];
	//if (idx <nspts) { printf("%d, %d \n", idx, SPptr[idx].cellnumber); }
	if (tidx < 27) { p[tidx] = 0; }
	
	if (start[bidx] >= 0) {
		//if (bidx == 0) { printf("%d\n", start[bidx]); }
		///////////count and sort population of neighbour cells//////////////
		if (tidx < 27 && bidx+ nb[tidx] >= 0 && bidx + nb[tidx] < NUMCELLS && start[bidx + nb[tidx]] >= 0 && end[bidx + nb[tidx]] >= 0 && start[bidx + nb[tidx]] < nspts && 1 + end[bidx + nb[tidx]] - start[bidx + nb[tidx]] > 0 ) {
			p[tidx] = 1 + end[bidx + nb[tidx]] - start[bidx + nb[tidx]]; //count population of neighbour cells so we know how many threads to use
			pidx[tidx] = tidx;
			
		}
		if (tidx == 13) { blockpop = p[tidx]; }
	}
	else {
		if (tidx == 13) { blockpop = 0; }
	}
	__syncthreads();


	//if (bidx == 21641 && tidx==0) { printf("%d %d %d \n", p[13], nb[13], start[nb[13]]); }
	if (start[bidx] >= 0) {
		if (tidx == 0) {
			total = 0;
			for (int i = 0; i < 27; i++) {
				
				if (p[i] < 64 && p[i]>0 && bidx + nb[i] >= 0 && bidx + nb[i] < NUMCELLS && start[bidx + nb[i]] >= 0 && end[bidx + nb[i]] >= 0 && start[bidx + nb[i]] < nspts) { total += p[i];	}
			
			}
		}
	}
	else {
		if (tidx == 0) {total = 0; }
	}
	
	__syncthreads();

	if (start[bidx] >= 0) {
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
	
	if (start[bidx] >= 0) {
		if (tidx < total) {
			sum[tidx] = 0;
			jj[tidx] = 0;
			while (tidx + 1 > sum[tidx]) {
				sum[tidx] += p[jj[tidx]];
				jj[tidx]++;
			}
		}
	}
	__syncthreads();
	
	//if (bidx== 34624 && tidx < total) { printf("tidx: %d, cell#:%d, jj:%d, sum:%d \n", tidx, bidx + nb[pidx[jj[tidx] - 1]], jj[tidx],p[jj[tidx]]); }
	//__syncthreads();
//	__shared__ float k[8];
//	__shared__ float rabx[8];
//	__shared__ float raby[8];
//	__shared__ float rabz[8];
//	__shared__ float vabx[8];
//	__shared__ float vaby[8];
//__shared__ float vabz[8];
	if (start[bidx] >= 0) {
		if (tidx < total && bidx + nb[pidx[jj[tidx] - 1]] >= 0 && bidx + nb[pidx[jj[tidx] - 1]] < NUMCELLS) {

			///////////////////////////////////////////////////////////////////
			volatile int j = start[bidx + nb[pidx[jj[tidx] - 1]]] + sum[tidx] - (tidx + 1);
			
			
			if (start[bidx + nb[pidx[jj[tidx] - 1]]] >= 0 && j < nspts && j >= 0) {
				//int i = start[bidx] + tidx / total;
				
				for (volatile int i = start[bidx]; i <= end[bidx]; i++) {
					float ds = (SPptr[i]).distance((SPptr[j]));
					if (ds <= (2 * cutoff) && ds > 0) {

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
						volatile float s = (ALPHA_FLUID * SOUND * (cutoff * (d / (d2 + 0.01*powf(cutoff, 2))) + 50 * 1.0 / SOUND*powf(cutoff * (d / (d2 + 0.01*powf(cutoff, 2))), 2)) / (((SPptr[i]).dens + (SPptr[j]).dens) / 2.0)) *(d < 0)*(1 + (!(SPptr[i]).boundary)*((SPptr[j]).boundary) * ALPHA_BOUNDARY);
						//float s2 = ALPHA_LAMINAR_FLUID * SOUND * cutoff / ((SPptr[i]).dens + (SPptr[j]).dens)*d*(d < 0) / (d2 + 0.01*pow(cutoff, 2))*(1 + (!(SPptr[i]).boundary)*((SPptr[j]).boundary) *ALPHA_LAMINAR_BOUNDARY); //laminar

						volatile float dpx = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dkx;
						volatile float dpy = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dky;
						volatile float dpz = ((SPptr[j]).press / powf((SPptr[j]).dens, 2) + (SPptr[i]).press / powf((SPptr[i]).dens, 2) + s)*dkz;

						//(SPptr[i]).vel_grad[0][0] += -vabx*dkxtest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[0][1] += -vaby*dkxtest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[0][2] += -vabz*dkxtest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[1][0] += -vabx*dkytest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[1][1] += -vaby*dkytest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[1][2] += -vabz*dkytest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[2][0] += -vabx*dkztest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[2][1] += -vaby*dkztest / (SPptr[i]).dens;
						//(SPptr[i]).vel_grad[2][2] += -vabz*dkztest / (SPptr[i]).dens;

						///(SPptr[i]).stress_accel[0] += ((SPptr[i]).stress_tensor[0][0] * dkxtest + (SPptr[i]).stress_tensor[0][1] * dkytest + (SPptr[i]).stress_tensor[0][2] * dkztest) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[0][0] * dkxtest + (SPptr[i]).stress_tensor[0][1] * dkytest + (SPptr[i]).stress_tensor[0][2] * dkztest) / pow((SPptr[i]).dens, 2);
						///(SPptr[i]).stress_accel[1] += ((SPptr[i]).stress_tensor[1][0] * dkxtest + (SPptr[i]).stress_tensor[1][1] * dkytest + (SPptr[i]).stress_tensor[1][2] * dkztest) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[1][0] * dkxtest + (SPptr[i]).stress_tensor[1][1] * dkytest + (SPptr[i]).stress_tensor[1][2] * dkztest) / pow((SPptr[i]).dens, 2);
						///(SPptr[i]).stress_accel[2] += ((SPptr[i]).stress_tensor[2][0] * dkxtest + (SPptr[i]).stress_tensor[2][1] * dkytest + (SPptr[i]).stress_tensor[2][2] * dkztest) / pow((SPptr[i]).dens, 2) + ((SPptr[i]).stress_tensor[2][0] * dkxtest + (SPptr[i]).stress_tensor[2][1] * dkytest + (SPptr[i]).stress_tensor[2][2] * dkztest) / pow((SPptr[i]).dens, 2);

						atomicAdd(&(SPptr[i].newdens), k *(1 + float(!(SPptr[i]).boundary)*float((SPptr[j]).boundary)*BDENSFACTOR));
						atomicAdd(&(SPptr[i].newdelpressx), dpx);
						atomicAdd(&(SPptr[i].newdelpressy), dpy);
						atomicAdd(&(SPptr[i].newdelpressz), dpz);
						__syncthreads();
					}
				}
			}
		}
	}

	/*
		float tempdens = 0;
		float tempdelpressx = 0;
		float tempdelpressy = 0;
		float tempdelpressz = 0;	
		//float tempdiffusionx = 0;
		//float tempdiffusiony = 0;
		//float tempdiffusionz = 0;
		

		if (idx<nspts){
			for (int i = 0; i < nspts; i++) {
				//if (idx != i && SPptr[idx].cellnumber == SPptr[i].cellnumber) { printf("%d, %d, %d \n", SPptr[idx].cellnumber, SPptr[i].cellnumber,neighbours[SPptr[idx].cellnumber*nspts + i]); }
				if (neighbours[SPptr[idx].cellnumber*nspts + i]) {
					//printf("%d, %d \n", SPptr[idx].cellnumber, SPptr[i].cellnumber);
					float ds = (SPptr[idx]).distance((SPptr[i]));
					
					if (ds <= (2 * cutoff) && ds > 0) {
						float k = kernel(ds);


						float rabx = (SPptr[idx]).rab_x((SPptr[i]));
						float raby = (SPptr[idx]).rab_y((SPptr[i]));
						float rabz = (SPptr[idx]).rab_z((SPptr[i]));
						float vabx = (SPptr[idx]).vab_x((SPptr[i]));
						float vaby = (SPptr[idx]).vab_y((SPptr[i]));
						float vabz = (SPptr[idx]).vab_z((SPptr[i]));
						float dkx = kernel_derivative(ds)*rabx / ds;
						float dky = kernel_derivative(ds)*raby / ds;
						float dkz = kernel_derivative(ds)*rabz / ds;

						float dkxtest = kernel_test(ds)*rabx / ds;
						float dkytest = kernel_test(ds)*raby / ds;
						float dkztest = kernel_test(ds)*rabz / ds;

						float d = dot_prod(vabx, vaby, vabz, rabx, raby, rabz);
						float d2 = pow(ds, 2);
						float s = (ALPHA_FLUID * SOUND * (cutoff * (d / (d2 + 0.01*pow(cutoff, 2))) + 50 * 1.0 / SOUND*pow(cutoff * (d / (d2 + 0.01*pow(cutoff, 2))), 2)) / (((SPptr[idx]).dens + (SPptr[i]).dens) / 2.0)) *(d < 0)*(1 + (!(SPptr[idx]).boundary)*((SPptr[i]).boundary) * ALPHA_BOUNDARY);
						float s2 = ALPHA_LAMINAR_FLUID * SOUND * cutoff / ((SPptr[idx]).dens + (SPptr[i]).dens)*d*(d < 0) / (d2 + 0.01*pow(cutoff, 2))*(1 + (!(SPptr[idx]).boundary)*((SPptr[i]).boundary) *ALPHA_LAMINAR_BOUNDARY); //laminar

						float dpx = ((SPptr[i]).press / pow((SPptr[i]).dens, 2) + (SPptr[idx]).press / pow((SPptr[idx]).dens, 2) + s + s2)*dkx;
						float dpy = ((SPptr[i]).press / pow((SPptr[i]).dens, 2) + (SPptr[idx]).press / pow((SPptr[idx]).dens, 2) + s + s2)*dky;
						float dpz = ((SPptr[i]).press / pow((SPptr[i]).dens, 2) + (SPptr[idx]).press / pow((SPptr[idx]).dens, 2) + s + s2)*dkz;

						//(SPptr[index]).vel_grad[0][0] += -vabx*dkxtest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[0][1] += -vaby*dkxtest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[0][2] += -vabz*dkxtest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[1][0] += -vabx*dkytest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[1][1] += -vaby*dkytest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[1][2] += -vabz*dkytest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[2][0] += -vabx*dkztest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[2][1] += -vaby*dkztest / (SPptr[i]).dens;
						//(SPptr[index]).vel_grad[2][2] += -vabz*dkztest / (SPptr[i]).dens;

						///(SPptr[index]).stress_accel[0] += ((SPptr[index]).stress_tensor[0][0] * dkxtest + (SPptr[index]).stress_tensor[0][1] * dkytest + (SPptr[index]).stress_tensor[0][2] * dkztest) / pow((SPptr[index]).dens, 2) + ((SPptr[i]).stress_tensor[0][0] * dkxtest + (SPptr[i]).stress_tensor[0][1] * dkytest + (SPptr[i]).stress_tensor[0][2] * dkztest) / pow((SPptr[i]).dens, 2);
						///(SPptr[index]).stress_accel[1] += ((SPptr[index]).stress_tensor[1][0] * dkxtest + (SPptr[index]).stress_tensor[1][1] * dkytest + (SPptr[index]).stress_tensor[1][2] * dkztest) / pow((SPptr[index]).dens, 2) + ((SPptr[i]).stress_tensor[1][0] * dkxtest + (SPptr[i]).stress_tensor[1][1] * dkytest + (SPptr[i]).stress_tensor[1][2] * dkztest) / pow((SPptr[i]).dens, 2);
						///(SPptr[index]).stress_accel[2] += ((SPptr[index]).stress_tensor[2][0] * dkxtest + (SPptr[index]).stress_tensor[2][1] * dkytest + (SPptr[index]).stress_tensor[2][2] * dkztest) / pow((SPptr[index]).dens, 2) + ((SPptr[i]).stress_tensor[2][0] * dkxtest + (SPptr[i]).stress_tensor[2][1] * dkytest + (SPptr[i]).stress_tensor[2][2] * dkztest) / pow((SPptr[i]).dens, 2);

						tempdens += k*(1 + float(!(SPptr[idx]).boundary)*float((SPptr[i]).boundary)*BDENSFACTOR);
						tempdelpressx += dpx;
						tempdelpressy += dpy;
						tempdelpressz += dpz;
						///tempdiffusionx += 1 / (SPptr[i]).dens*dkx;
						///tempdiffusiony += 1 / (SPptr[i]).dens*dky;
						///tempdiffusionz += 1 / (SPptr[i]).dens*dkz;
					}
				}
			}


		(SPptr[idx]).newdens = (tempdens);
		(SPptr[idx]).newdelpressx = tempdelpressx;
		(SPptr[idx]).newdelpressy = tempdelpressy;
		(SPptr[idx]).newdelpressz = tempdelpressz;
		//(SPptr[idx]).diffusionx = tempdiffusionx;
		//(SPptr[idx]).diffusiony = tempdiffusiony;
		//(SPptr[idx]).diffusionz = tempdiffusionz;

		/*if ((SPptr[index]).solid) {
			float tr = 0; //trace of strain rate
			float tr2 = 0; //trace of stress tensor
			float tr3 = 0; //double dot of stress tensor
			float tr4 = 0; //trace of stress tensor times strain rate
			float tr5 = 0; //double dot of strain rate
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					(SPptr[index]).strain_rate[p][q] = 0.5*((SPptr[index]).vel_grad[p][q] + (SPptr[index]).vel_grad[q][p]);
					(SPptr[index]).stress_tensor_squared[p][q] = pow((SPptr[index]).stress_tensor[p][q], 2);
					tr3 += 0.5*(SPptr[index]).stress_tensor_squared[p][q];
					(SPptr[index]).strain_rate_squared[p][q] = pow((SPptr[index]).strain_rate[p][q], 2);
					tr5 += (SPptr[index]).strain_rate_squared[p][q];
					tr4 += (SPptr[index]).stress_tensor[p][q] * (SPptr[index]).strain_rate[q][p];
				}
				tr += (SPptr[index]).strain_rate[p][p];
				tr2 += (SPptr[index]).stress_tensor[p][p];



			}

			//	std::cout << (SPptr[index]).press << "\n";
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					if (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[index]).press + KC / (sqrt(9 + 12 * pow(tan(PHI), 2))) < tr3 && tr3 != 0) {
						(SPptr[index]).stress_tensor[p][q] *= (3 * tan(PHI) / (sqrt(9 + 12 * pow(tan(PHI), 2)))*(SPptr[index]).press + KC / (sqrt(9 + 12 * pow(tan(PHI), 2)))) / tr3;
					}
					(SPptr[index]).stress_rate[p][q] = 3 * C1*((SPptr[index]).press)*((SPptr[index]).strain_rate[p][q] - 1. / 3.*tr*(p == q)) + C1*C2*(tr4 + tr*(SPptr[index]).press) / (pow((SPptr[index]).press, 2) + 1e8)*(SPptr[index]).stress_tensor[p][q] - C1*C3*sqrt(tr5)*(SPptr[index]).stress_tensor[p][q];
					//std::cout << tr4 << ", " << tr*(SPptr[index]).press << "\n";

				}
			}
		}*/
		//}			
		__syncthreads();
		
	}

__global__ void mykernel2(Particle *SPptr, int *cells, int *start, int *end, int nspts, float *spts, float *a3, float *b3) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	if (index < nspts) {
		if (!(SPptr[index]).flag) {
			spts[(3 * index)] = (SPptr[index]).xcoord;
			spts[(3 * index) + 1] = (SPptr[index]).ycoord;
			spts[(3 * index) + 2] = (SPptr[index]).zcoord;
			a3[index] = ((SPptr[index])).dens;
			b3[index] = SPptr[index].cellnumber;
			
		}

		(SPptr[index]).update();
		(SPptr[index]).cellnumber = int((SPptr[index].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[index].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[index].zcoord - ZMIN) / CELLSIZE);
		//SPptr[index].cellnumber = morton(int((SPptr[index].xcoord - XMIN) / CELLSIZE), int((SPptr[index].ycoord - YMIN) / CELLSIZE), int((SPptr[index].zcoord - ZMIN) / CELLSIZE));
		cells[index] = SPptr[index].cellnumber;
		SPptr[index].newdens = 0;
		SPptr[index].newdelpressx = 0;
		SPptr[index].newdelpressy = 0;
		SPptr[index].newdelpressz = 0;
	}
	if (index < NUMCELLS) {
		start[index] = -1;
		end[index] = -1;
	}
	__syncthreads();
}

		

	