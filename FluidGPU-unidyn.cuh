#define XMIN -1
#define YMIN -1
#define ZMIN -1
#define XMAX 1
#define YMAX 1
#define ZMAX 1
#define CELLSIZE 0.12
#define GRIDSIZE 17 //Should be (XMAX - XMIN) / CELLSIZE
#define NUMCELLS 4913//39304//4913//Should be GRIDSIZE cubed
#define GRAVITY -9.8 // gravity acceleration m/s^2
#define SOUND 1450.0 // speed of sound in m/s
#define RHO_0 9550 //reference density of water in kg/m^3
#define RHO_0_SAND 9550
#define P_0 101325 //reference pressure of water in Pa
#define DIFF 0//0.0000001 //diffusion magnitude

#define ALPHA_FLUID -0.0155e1  //-.155 for dt 001
#define ALPHA_BOUNDARY 80e0 // (should be high to prevent penetration) 80 at dt 001

#define ALPHA_SAND -0.0155e2  //1.55 for sand viscosity
#define ALPHA__SAND_BOUNDARY 100e-1 //10 for sand viscosity for boundary (should be high to prevent penetration)


#define BDENSFACTOR 1.5 //Density is increased in boundary particles

#define C1 1.5e1  //stress tensor constants for granular material
#define C2 0e6//1e3
#define C3 5e1//1e0
#define PHI 1.23 //friction angle (radians)
#define KC 1e9 //cohesion

#define MIXPRESSURE 1e-12
#define MIXBROWNIAN 5e-9

#define cutoff 0.06
#define DT 0.0018 //Time step size
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
    

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					system("pause"); \
					exit(1);															\
		} }

__host__ __device__ float kernel(float r);

__host__ __device__ float kernel_test(float r);

__host__ __device__ float kernel_derivative(float r);

//Dot product
__host__ __device__ inline float dot_prod(float x1, float y1, float z1, float x2, float y2, float z2);
//Cross products
__host__ __device__ inline float cross_prod_x(float x1, float y1, float z1, float x2, float y2, float z2);

__host__ __device__ inline float cross_prod_y(float x1, float y1, float z1, float x2, float y2, float z2);

__host__ __device__ inline float cross_prod_z(float x1, float y1, float z1, float x2, float y2, float z2);


class Particle {

public:

	//Constructor
	__host__ __device__ Particle() {
		xcoord = ycoord = zcoord = 0.;
		xvel = yvel = zvel = 0.;
		xacc = yacc = 0.;
		zacc = GRAVITY;
		flag = false;
		boundary = false;
	}

	__host__ __device__ ~Particle() {}

	//Constructor
	__host__ __device__ Particle(float x, float y, float z) {
		xcoord = x;
		ycoord = y;
		zcoord = z;
		xvel = yvel = zvel = 0;
		xacc = yacc = 0;
		zacc = GRAVITY;
		flag = false;
		boundary = false;
	}

	//Constructor for boundary particles
	__host__ __device__ Particle(float x, float y, float z, bool b) {
		xcoord = x;
		ycoord = y;
		zcoord = z;
		xvel = yvel = zvel = 0;
		xacc = yacc = zacc = 0;
		boundary = b;
		flag = false;
	}

	//Constructor
	__host__ __device__ Particle(float x, float y, float z, float vx, float vy, float vz) {
		xcoord = x;
		ycoord = y;
		zcoord = z;
		xvel = vx;
		yvel = vy;
		zvel = vz;
		xacc = yacc = 0;
		zacc = GRAVITY;
		flag = false;
		boundary = false;
	}

	//Coordinates
	float xcoord;
	float ycoord;
	float zcoord;

	//Velocity
	float xvel;
	float yvel;
	float zvel;

	float xvel_prev=0;
	float yvel_prev=0;
	float zvel_prev=0;

	//Acceleration
	float xacc;
	float yacc;
	float zacc;

	float xacc_prev=0;
	float yacc_prev=0;
	float zacc_prev=0;

	//Index for tracking
	int index;
    int cellnumber;
    int subindex;

	//Physical parameters
	float mass = 1;
	float dens = RHO_0; //Default density water
	float press = 0; //Default pressure of 101325 Pa (kg/s^2/m) = 1 atm
	float delpressz = 0;
	float delpressy = 0;
	float delpressx = 0;
	float diffusionx = 0;
	float diffusiony = 0;
	float diffusionz = 0;
	//float sigma = 0;


	float newdens = RHO_0; //Default density water
	float newdelpressz =0;
	float newdelpressy = 0;
	float newdelpressx = 0;
	//float newsigma = 0;
	float soliddriftvelx = 0;
	float soliddriftvely = 0;
	float soliddriftvelz = 0;
	float fluiddriftvelx = 0;
	float fluiddriftvely = 0;
	float fluiddriftvelz = 0;
	//float friction = 0;

	float vel_grad[3][3] = { {0} }; //velocity gradient field
	float strain_rate[3][3] = { {0} };
	float stress_rate[3][3] = { {0} };
	float stress_tensor[3][3] = { {0} };
	float stress_accel[3] = { 0 };
	float mixture_accel[3] = { 0 };

	bool boundary;
	float solid = 0;
	float fluid = 1;
	float delsolid = 0;
	float delfluid = 0;
	bool flag;
	bool split = false;
	

	__host__ __device__ void set_dens(float x) {
		dens = (x + kernel(0)) / 23.0 *(1 + float(boundary)*BDENSFACTOR) + 9250;
	}

	__host__ __device__ void set_delpress(float x, float y, float z) {
		delpressx = x;
		delpressy = y;
		delpressz = z;
	}

	__host__ __device__ void set_coord(float x, float y, float z) {
		xcoord = x;
		ycoord = y;
		zcoord = z;
	}

	__host__ __device__ void set_vel(float x, float y, float z) {
		xvel = x;
		yvel = y;
		zvel = z;
	}

	__host__ __device__ void set_flag() {
		flag = true;
	}


	//Calculate distance between particles
	 __device__ inline float distance(Particle P) {
		return sqrt(powf(rab_x(P), 2) + powf(rab_y(P), 2) + powf(rab_z(P), 2));
	}

	__device__ float rab_x(Particle P) {
		return xcoord - P.xcoord;
	}

	__device__ float rab_y(Particle P) {
		return ycoord - P.ycoord;
	}

	__device__ float rab_z(Particle P) {
		return zcoord - P.zcoord;
	}

	__device__ float vab_x(Particle P) {
		return xvel - P.xvel;
	}

	__device__ float vab_y(Particle P) {
		return yvel - P.yvel;
	}

	__device__ float vab_z(Particle P) {
		return zvel - P.zvel;
	}

	//Calculate point density from particle pair
	 __device__ float density(Particle P) {
		 return kernel(distance(P))*(1 - P.solid) + P.solid*kernel(distance(P))*RHO_0_SAND / RHO_0;
	}

	 __device__ float kernel_derivative_x(Particle P) {
		return kernel_derivative(distance(P))* rab_x(P) / distance(P);
	}

	 __device__ float kernel_derivative_y(Particle P) {
		return kernel_derivative(distance(P))* rab_y(P) / distance(P);
	}

	 __device__ float kernel_derivative_z(Particle P) {
		return kernel_derivative(distance(P))* rab_z(P) / distance(P);
	}

	//Calculate contribution to del of pressure due from one neighbour particle
	 __device__ float del_pressure_mag(Particle P) {
		return (P.press / powf(P.dens, 2) + press / pow(dens, 2))*kernel_derivative(distance(P)); //in direction r1-r2
	}

	 __device__ float del_pressure_x(Particle P) {
		return ((P.press / powf(P.dens, 2) + press / pow(dens, 2) + calculate_sigma(P))*kernel_derivative_x(P)); //x-component
	}

	 __device__ float del_pressure_y(Particle P) {
		return ((P.press / powf(P.dens, 2) + press / powf(dens, 2) + calculate_sigma(P))*kernel_derivative_y(P)); //y-component
	}

	 __device__ float del_pressure_z(Particle P) {
		return ((P.press / powf(P.dens, 2) + press / powf(dens, 2) + calculate_sigma(P))*kernel_derivative_z(P)); //z-component
	}

	//calculate pressure from current density
	 __device__ void calculate_pressure(void) {
		 press = (1 - solid) * 1000 * pow(SOUND, 0)*RHO_0 / 7.0*(pow(dens / RHO_0, 7) - 1) + (solid) * 1000 * pow(SOUND, 0)*RHO_0_SAND / 7.0*(pow(dens / RHO_0, 7) - 1);   //b*[(rho/rho0)^gamma -1]     gamma = 7, ref density = 1, b = speed of sound in medium squared at reference density
																			   //press*= (press >= 0);
																			   //press = 101325 * (pow(dens / RHO_0, 7));// *pow(10, 30);
	}

	 __device__ float calculate_sigma(Particle P) {
		float d = dot_prod(vab_x(P), vab_y(P), vab_z(P), rab_x(P), rab_y(P), rab_z(P));     //gamma*(mean speed of sound)cutoff*(v_1-v_2 dot r_1-r_2)/((r_1-r_2)^2+0.01 cutoff^2) / (mean density)
		float d2 = powf(distance(P), 2);
		return (ALPHA_FLUID * SOUND * cutoff * (d / (d2 + 0.01*powf(cutoff, 2))) / ((dens + P.dens) / 2.0)) *(d < 0)*(1 + !boundary*P.boundary*ALPHA_BOUNDARY);
		// alpha = 0.01 but may need to be tuned depending on cutoff

	}

	 __device__ void update(int t) {

		if (!flag) { //check if particle has already been updated

			set_dens(newdens);
			calculate_pressure();
			set_delpress(newdelpressx, newdelpressy, newdelpressz);

			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					stress_tensor[p][q] = DT*stress_rate[p][q];
				}
			}

			if (!boundary) {
				volatile float friction = abs(diffusionx) + abs(diffusiony) + abs(diffusionz);
				solid += DT*delsolid;
				solid *= (solid >= 0.0);

				if (fluid + delfluid < 0.2) { delfluid = 0; }
				fluid += DT*delfluid;
				fluid *= (fluid >= 0);

				fluid *= 1 / (fluid + solid);
				solid *= 1 / (fluid + solid);
				
				//euler
				//xcoord = xcoord + DT*xvel + DIFF*diffusionx;
				//ycoord = ycoord + DT*yvel + DIFF*diffusiony;
				//zcoord = zcoord + DT*zvel + DIFF*diffusionz;

				//leapfrog
				xcoord = xcoord + DT*xvel + 0.5*DT*DT*xacc + DIFF*diffusionx;
				ycoord = ycoord + DT*yvel + 0.5*DT*DT*yacc +DIFF*diffusiony;
				zcoord = zcoord + DT*zvel + 0.5*DT*DT*zacc +DIFF*diffusionz;

				if (zcoord < -0.89){
					//curandState_t state;
					//curand_init(cellnumber, 0, 0, &state);
					//int result = curand(&state) % 100;
					//zcoord += 1.6;
					xvel = 0;
					yvel = 0;
					//xcoord = cosf(xcoord*313)/3.0;
					//ycoord = cosf(ycoord*313)/3.0+0.3;
				}
				//euler
				//xvel = (xvel + DT*xacc + DT*(stress_accel[0]) + 5 * DT*DT*(mixture_accel[0])) - ((xvel + DT*xacc + DT*(stress_accel[0])+ DT*DT*(mixture_accel[0])) > 0)*friction*0.0000002*solid + ((xvel + DT*xacc + DT*(stress_accel[0]) +  DT*DT*(mixture_accel[0])) < 0)*friction *0.0000002*solid;
				//xvel *= (abs(xvel) > solid*0.002);
				//yvel = (yvel + DT*yacc + DT*(stress_accel[1]) + 5 * DT*DT*(mixture_accel[1])) - ((yvel + DT*yacc + DT*(stress_accel[1])+ DT*DT*(mixture_accel[1])) > 0)*friction*0.0000002*solid + ((yvel + DT*yacc + DT*(stress_accel[1]) +  DT*DT*(mixture_accel[1])) < 0)*friction *0.0000002*solid;
				//yvel *= (abs(yvel) > solid*0.002);
				//zvel = (zvel + DT*zacc + DT*(stress_accel[2]) + 5 * DT*DT*(mixture_accel[2])) - ((zvel + DT*yacc + DT*(stress_accel[2]) + DT*DT*(mixture_accel[2])) > 0)*friction*0.0000002*solid + ((zvel + DT*yacc + DT*(stress_accel[2]) + DT*DT*(mixture_accel[2])) < 0)*friction *0.0000002*solid;
				//zvel *= (abs(zvel) > solid*0.002);

				//leapfrog
				xvel = (xvel + 0.5*DT*xacc + DT*(stress_accel[0]) + 5 * DT*DT*(mixture_accel[0])) - ((xvel + DT*xacc + DT*(stress_accel[0])+ DT*DT*(mixture_accel[0])) > 0)*friction*0.0000002*solid + ((xvel + DT*xacc + DT*(stress_accel[0]) +  DT*DT*(mixture_accel[0])) < 0)*friction *0.0000002*solid;
				yvel = (yvel + 0.5*DT*yacc + DT*(stress_accel[1]) + 5 * DT*DT*(mixture_accel[1])) - ((xvel + DT*xacc + DT*(stress_accel[1])+ DT*DT*(mixture_accel[1])) > 0)*friction*0.0000002*solid + ((xvel + DT*xacc + DT*(stress_accel[1]) +  DT*DT*(mixture_accel[1])) < 0)*friction *0.0000002*solid;
				zvel = (zvel + 0.5*DT*zacc + DT*(stress_accel[2]) + 5 * DT*DT*(mixture_accel[2])) - ((xvel + DT*xacc + DT*(stress_accel[2])+ DT*DT*(mixture_accel[2])) > 0)*friction*0.0000002*solid + ((xvel + DT*xacc + DT*(stress_accel[2]) +  DT*DT*(mixture_accel[2])) < 0)*friction *0.0000002*solid;
				

				//Acceleration due to grav
				xacc = -((220.0-70.0*solid) / dens)*delpressx;
				yacc = -((220.0 - 70.0 * solid) / dens)*delpressy;
				zacc = GRAVITY + ((-220.0 +70.0*solid)/ dens)*delpressz; //220 at DT 001
				
				//Runge-kutta
				/*xvel += DT*xacc;
				yvel += DT*yacc;
				zvel += DT*zacc;
				if (t%2 ==0){
					xacc_prev=xacc;
					yacc_prev=yacc;
					zacc_prev=zacc;
					xvel_prev = xvel;
					yvel_prev = yvel;
					zvel_prev = zvel;
				
					xcoord += DT*xvel;
					ycoord += DT*yvel;
					zcoord += DT*zvel;
				}	
				if (t%2 ==1){			
					xcoord += DT/2.0*(-xvel_prev+xvel);
					ycoord += DT/2.0*yvel*(-yvel_prev+yvel);
					zcoord += DT/2.0*zvel*(-zvel_prev+zvel);
					xvel += DT/2.0*(-xacc_prev+xacc);
					yvel += DT/2.0*(-yacc_prev+yacc);
					zvel += DT/2.0*(-zacc_prev+zacc);
				}
				if (zcoord < -0.89){
					zcoord += 1.6;
				}*/

				//leapfrog
				xvel +=0.5*xacc*DT;
				yvel +=0.5*yacc*DT;
				zvel +=0.5*zacc*DT;

				if (abs(zvel) > 7.5){
					//zvel = 7.5/zvel;
				}
				if (abs(yvel) > 7.0){
					//yvel = 7.0/yvel;
				}
				if (abs(xvel) > 7.0){
					//xvel = 7.0/xvel;
				}

				if (abs(zcoord) > 0.98){
					zcoord = 0.97/zcoord;
					zvel = 0;//-zvel;
				}
				if (abs(ycoord) > 0.98){
					yvel = -yvel;
				}
				if (abs(xcoord) > 0.98){
					xvel = -xvel;
				}

			}
			else{
				//zvel = acosf(abs(50-((t/2)%100))/50.1)-1.0;
				//zcoord +=2.5*zvel*DT;
				//printf("%4.4f, %4.4f\n", float(t), zvel);
			}
		}
		flag = true; //reset flag for next timestep
	}
};

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


//DEFINE NODES///////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


struct node
{
	Particle data;
	node *next;
};

class list
{
	//private:
	//node *head, *tail;
public:
	node *head, *tail;
	list()
	{
		head = NULL;
		tail = NULL;
	}

	__host__ __device__ void createnode(Particle &value)
	{
		node *temp = new node;
		temp->data = value;
		temp->next = NULL;
		if (head == NULL)
		{
			head = temp;
			tail = temp;
			temp = NULL;
		}
		else
		{
			tail->next = temp;
			tail = temp;
		}
	}

	__host__ __device__ void insert_start(Particle &value)
	{
		node *temp = new node;
		temp->data = value;
		temp->next = head;
		head = temp;
	}

	__host__ __device__ void insert_position(int pos, Particle value)
	{
		node *pre = new node;
		node *cur = new node;
		node *temp = new node;
		cur = head;
		for (int i = 1; i<pos; i++)
		{
			pre = cur;
			cur = cur->next;
		}
		temp->data = value;
		pre->next = temp;
		temp->next = cur;
	}

	__host__ __device__ void delete_first()
	{
		node *temp = new node;
		temp = head;
		head = head->next;
		delete temp;
	}

	__host__ __device__ void delete_last()
	{
		node *current = new node;
		node *previous = new node;
		current = head;
		while (current->next != NULL)
		{
			previous = current;
			current = current->next;
		}
		tail = previous;
		previous->next = NULL;
		delete current;
	}

	__host__ __device__ void delete_position(int pos)
	{
		node *current = new node;
		node *previous = new node;
		current = head;

		if (pos == 1) {
			head = head->next;
			delete current;
		}

		for (int i = 1; i<pos; i++)
		{
			previous = current;
			current = current->next;
		}
		previous->next = current->next;
	}
};

__global__ void findneighbours(int *cell, int *start, int *start_copy, int *end,  int nspts, int x);
__global__ void mykernel(Particle *SPptr, int *particleindex, int *cell, int *start, int *end, int *split, int nspts,int x,int dev, int buffer,int *numsplit);
__global__ void mykernel3(Particle *SPptr, int *particleindex, int *cell, int *start, int *end, int *split, int nspts,int x,int dev, int buffer,int *numsplit);
__global__ void mykernel2(Particle *SPptr, int *particleindex, int *cell, int *start_copy, int *start, int *end, int *split, int *numsplit, int nspts, int x,int dev, int buffer, int t, float *spts, float *a3, float *b3);
__global__ void find_idx(int *SPptr, int dev, int npts, int buffer, int *xleft, int *xright, int *sleft, int *sright);
__global__ void mem_shift(Particle *SPptr, Particle *buff, int *cells, int *ibuff, int dev, int shifts, int indexleft, int indexright);
__global__ void cell_calc(Particle *SPptr, int *particleindex, int *cells, int size, int dev) ;
__global__ void count_after_merge(int *cells, int *particleindex, int size, int *newsize);