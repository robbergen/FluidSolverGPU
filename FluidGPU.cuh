#define XMIN -1
#define YMIN -1
#define ZMIN -1
#define XMAX 1
#define YMAX 1
#define ZMAX 1
#define CELLSIZE 0.05
#define GRIDSIZE 40 //Should be (XMAX - XMIN) / CELLSIZE
#define NUMCELLS 64000//Should be GRIDSIZE cubed
#define GRAVITY -9.8 // gravity acceleration m/s^2
#define SOUND 1450.0 // speed of sound in m/s
#define RHO_0 9550 //reference density of water in kg/m^3
#define P_0 101325 //reference pressure of water in Pa
#define DIFF 0//0.0000001 //diffusion magnitude

#define ALPHA_FLUID -0.01e2  //viscosity for fluid
#define ALPHA_BOUNDARY 2000e-1 //viscosity for boundary (should be high to prevent penetration)

#define ALPHA_LAMINAR_FLUID -1.0e0  //viscosity for fluid
#define ALPHA_LAMINAR_BOUNDARY 0e0 //viscosity for boundary (should be high to prevent penetration)

#define BDENSFACTOR 1.5 //Density is increased in boundary particles

#define C1 1.5e1  //stress tensor constants for granular material
#define C2 -0e5//1e3
#define C3 -0e3//1e0
#define PHI 1.23 //friction angle (radians)
#define KC 1e3 //cohesion

#define cutoff 0.06
#define DT 0.0005 //Time step size
#include <cuda_runtime.h>

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

	//Acceleration
	float xacc;
	float yacc;
	float zacc;

	//Index for tracking
	int index;
	int cellnumber;

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
	float sigma = 0;


	float newdens = RHO_0; //Default density water
	float newpress = P_0; //Default pressure of 101325 Pa (kg/s^2/m) = 1 atm
	float newdelpressz =0;
	float newdelpressy = 0;
	float newdelpressx = 0;
	float newsigma = 0;

	float vel_grad[3][3] = { 0 }; //velocity gradient field
	float strain_rate[3][3] = { 0 };
	float stress_rate[3][3] = { 0 };
	float strain_rate_squared[3][3] = { 0 };
	float stress_tensor[3][3] = { 0 };
	float stress_tensor_squared[3][3] = { 0 };
	float stress_accel[3] = { 0 };


	bool boundary;
	bool solid = false;
	bool flag;


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
		return kernel(distance(P));
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
		press = 1000 * powf(SOUND, 0)*RHO_0 / 7.0*(powf(dens / RHO_0, 7) - 1);   //b*[(rho/rho0)^gamma -1]     gamma = 7, ref density = 1, b = speed of sound in medium squared at reference density
																			   //press*= (press >= 0);
																			   //press = 101325 * (pow(dens / RHO_0, 7));// *pow(10, 30);
	}

	 __device__ float calculate_sigma(Particle P) {
		float d = dot_prod(vab_x(P), vab_y(P), vab_z(P), rab_x(P), rab_y(P), rab_z(P));     //gamma*(mean speed of sound)cutoff*(v_1-v_2 dot r_1-r_2)/((r_1-r_2)^2+0.01 cutoff^2) / (mean density)
		float d2 = powf(distance(P), 2);
		return (ALPHA_FLUID * SOUND * cutoff * (d / (d2 + 0.01*powf(cutoff, 2))) / ((dens + P.dens) / 2.0)) *(d < 0)*(1 + !boundary*P.boundary*ALPHA_BOUNDARY);
		// alpha = 0.01 but may need to be tuned depending on cutoff

	}

	 __device__ void update(void) {

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

				xcoord = xcoord + DT*xvel + DIFF*diffusionx;
				ycoord = ycoord + DT*yvel + DIFF*diffusiony;
				zcoord = zcoord + DT*zvel + DIFF*diffusionz;

				xvel = (xvel + DT*xacc + DT*(stress_accel[0])) - ((xvel + DT*xacc + DT*(stress_accel[0]))>0)*0.003 + ((xvel + DT*xacc + DT*(stress_accel[0]))<0)*0.003;
				xvel *= (abs(xvel) > 0.003);
				yvel = (yvel + DT*yacc + DT*(stress_accel[1])) - ((yvel + DT*yacc + DT*(stress_accel[1]))>0)*0.003 + ((yvel + DT*yacc + DT*(stress_accel[1]))<0)*0.003;
				yvel *= (abs(yvel) > 0.003);
				zvel = (zvel + DT*zacc + DT*(stress_accel[2]));
				zvel *= (abs(zvel) > 0.003);

				//Acceleration due to grav
				xacc = -(150.0 / dens)*delpressx;
				yacc = -(150.0 / dens)*delpressy;
				zacc = GRAVITY + (-150.0 / dens)*delpressz;
			}
		}
		flag = false; //reset flag for next timestep
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
__global__ void findneighbours(int *cell, int *start, int *end,  int nspts);
__global__ void mykernel(Particle *SPptr, int *cell, int *start, int *end, int nspts);
__global__ void mykernel2(Particle *SPptr, int *cell, int *start, int *end, int nspts, float *spts, float *a3, float *b3);