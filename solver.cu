#include "visit_writer.h"
#include <math.h>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <algorithm>
#include <future>
#include <vector>
#include "FluidGPU.cuh"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

const int nspts = 8000; //number of solid particles
const int nbpts = 000;//1000; //number of solid particles
const int tpts = 4000;

//Storage for output
int vardims3[] = { 1,1 };
int morton_host(unsigned int x, unsigned int y, unsigned int z) {
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

int demorton_host(unsigned int x, int b) {
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

int main(int argc, char **argv)
{
	/*
	std::cout << morton(30, 30, 30) << "\n";
	for (int k = -1; k < 2; k++)
		for (int j = -1; j < 2; j++)
			for (int i = -1; i < 2; i++)
				std::cout << morton(demorton(morton(30 + i, 30 + j, 30 + k), 0), demorton(morton(30 + i, 30 + j, 30 + k), 1), demorton(morton(30 + i, 30 + j, 30 + k), 2)) << "\n";
				*/
	//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	int *pop, *d_pop, *popidx, *d_popidx;
	pop = (int *)malloc(sizeof(int)*27);
	popidx = (int *)malloc(sizeof(int) * 27);
	cudaMalloc((void **)&d_pop, sizeof(int) * 27);
	cudaMalloc((void **)&d_popidx, sizeof(int) * 27);
	for (int i = 0; i < 27; i++) { 	pop[i] = 0;	popidx[i] = i; }
	cudaMemcpy(d_pop, pop, sizeof(int)*27, cudaMemcpyHostToDevice);
	cudaMemcpy(d_popidx, popidx, sizeof(int) * 27, cudaMemcpyHostToDevice);

	size_t size = (nspts+nbpts) * sizeof(Particle);
	Particle *SPptr;
	Particle *d_SPptr;
	SPptr = (Particle *)malloc(size); // Allocate particles on host
	cudaMalloc((void **)&d_SPptr, size); // Allocate particles on device

	bool *neighbours;
	bool *d_neighbours;
	neighbours = (bool *)malloc(sizeof(int)*nspts*(NUMCELLS)); // Allocate particles on host
	cudaMalloc((void **)&d_neighbours, sizeof(bool) * nspts * (NUMCELLS)); // Allocate particles on device
	for (int i = 0; i < nspts * (NUMCELLS); i++)
		neighbours[i] = false;
	cudaMemcpy(d_neighbours, neighbours, sizeof(bool) * nspts * (NUMCELLS), cudaMemcpyHostToDevice);

	float *spts;
	cudaMallocManaged(&spts, 3*(nspts + nbpts) * sizeof(float));

	for (int i = 0; i < 3*(nspts + nbpts); i++)
		spts[i] =0 ;

	float *a3;
	cudaMallocManaged(&a3, (nspts + nbpts) * sizeof(float));
	float *b3;
	cudaMallocManaged(&b3, (nspts + nbpts) * sizeof(float));

	const char * const varnames3[] = { "dens", "cellnumber" };
	float *arrays3[] = { (float*)a3, (float*)b3, };


	
	//Set up Solid Particles
	
	for (int j = 0; j < nspts; j++) {
			SPptr[j] = *(new Particle(-.16 + 0.04*((j / 15) % 15), -0.76 + 0.04*(j / 15 / 15), -0.20 + (j % 15)*0.04, 0., 0., 0.));
			SPptr[j].index = j;
			SPptr[j].solid = true;
			SPptr[j].cellnumber = int((SPptr[j].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[j].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[j].zcoord - ZMIN) /CELLSIZE);
			//SPptr[j].cellnumber = morton_host(int((SPptr[j].xcoord - XMIN) / CELLSIZE), int((SPptr[j].ycoord - YMIN) / CELLSIZE), int((SPptr[j].zcoord - ZMIN)/CELLSIZE));
	}

	//Set up boundary particles
	for (int i = 0; i < nbpts; i++) {
		SPptr[nspts + i] = *(new Particle(-0.96 + 0.06*(i % 30), -0.96 + 0.06*(i / 30), -0.24, true));
		SPptr[i+nspts].index = nspts+i;
		SPptr[i + nspts].cellnumber = int((SPptr[i + nspts].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[i + nspts].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[i + nspts].zcoord - ZMIN) / CELLSIZE);
		//SPptr[i+nspts].cellnumber = morton_host(int((SPptr[i + nspts].xcoord - XMIN) / CELLSIZE), int((SPptr[i + nspts].ycoord - YMIN) / CELLSIZE), int((SPptr[i + nspts].zcoord - ZMIN) / CELLSIZE));
	}
	
	cudaMemcpy(d_SPptr, SPptr, size, cudaMemcpyHostToDevice);
	

	
	///////Sort particles by cell number and keep track of when a new cell starts//////////
	int *v_h, *v_d;
	//int *cellstart_h, *cellstart_d;
	const int N = nspts + nbpts;  // Number of elements in arrays
	size_t sizes = N * sizeof(int);
	//size_t sizes2 = 2*NUMCELLS * sizeof(int);
	v_h = (int *)malloc(sizes);        // Allocate array on host
	cudaMalloc((void **)&v_d, sizes);// Allocate array on device
	
	for (int i = 0; i<N; i++)
	{
		v_h[i] = SPptr[i].cellnumber;
		//std::cout << v_h[i] << "\n";
	}

	//cudaMemcpy(cellstart_d, cellstart_h, sizes2, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_h, sizes, cudaMemcpyHostToDevice);
	thrust::device_ptr<Particle> t_a(d_SPptr);
	thrust::device_ptr<int> t_v(v_d);
	cudaMemcpy(SPptr, d_SPptr, size, cudaMemcpyDeviceToHost);
	
	int *start, *end, *d_start, *d_end;
	size_t sizes2 = NUMCELLS * sizeof(int);
	start = (int *)malloc(sizes2);        // Allocate array on host
	end = (int *)malloc(sizes2);        // Allocate array on host
	cudaMalloc((void **)&d_start, sizes2);// Allocate array on device
	cudaMalloc((void **)&d_end, sizes2);// Allocate array on device

	for (int i = 0; i<NUMCELLS; i++)
	{
		start[i] = -1;
		end[i] = -1;
	}
	cudaMemcpy(d_start, start, sizes2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_end, end, sizes2, cudaMemcpyHostToDevice);

	for (int t = 0; t < tpts; t++) {
		std::cout << "t= " << t << "\n";
		

		cudaEvent_t start, stop;
		CUDA_CHECK_RETURN(cudaEventCreate(&start));
		CUDA_CHECK_RETURN(cudaEventCreate(&stop));
		float elapsedTime;
		CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
		cudaDeviceSynchronize();
		thrust::sort_by_key(t_v, t_v + N, t_a);
		findneighbours << <NUMCELLS, 1024 >> > (v_d, d_start, d_end, nspts+nbpts);

		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";

	
		mykernel <<<NUMCELLS, 64 >> > (d_SPptr, v_d, d_start, d_end, nspts+nbpts);
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

		CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
		CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaEventDestroy(start));
		CUDA_CHECK_RETURN(cudaEventDestroy(stop));
		std::cout << "done.\nElapsed kernel time: " << elapsedTime << " ms\n";
		mykernel2 <<<NUMCELLS, 1024>> > (d_SPptr, v_d, d_start, d_end, nspts + nbpts, spts, a3,b3);

		//std::cout << cudaGetErrorName(cudaGetLastError()) << "\n";
		

//		cudaDeviceSynchronize();

		
		if (t % 10 == 0) {
			cudaDeviceSynchronize();
			//Write each frame to file
			std::ostringstream oss;
			oss << "C:\\Users\\robbe\\Desktop\\Code\\anim_s" << t / 10 << ".vtk";
			std::string var = oss.str();
			const char* cstr = var.c_str();
			//write_point_mesh(cstr, 0, nspts + nbpts, spts, 2, vardims3, varnames3, arrays3);

		}
	}
	return 0;
}
