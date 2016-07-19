////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

////////////////////////////////////////////////////////////////////////////////
#define WA 1024
#define HA 1024
#define WB 1024

#define HB WA
#define WC WB
#define HC HA
////////////////////////////////////////////////////////////////////////////////


// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
	int i;

	for (i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename program filename
//! @param cPreamble code that is prepended to the loaded file, typically a set of
//! #defines or a header
//! @param szFinalLength returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t*
	szFinalLength)
{
	// locals
	FILE* pFileStream = NULL;
	size_t szSourceLength;
	// open the OpenCL source code file
	if (fopen_s(&pFileStream, cFilename, "rb") != 0)
	{
		return NULL;
	}
	size_t szPreambleLength = strlen(cPreamble);
	// get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);
	// allocate a buffer for the source code string and read it in
	char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
	memcpy(cSourceString, cPreamble, szPreambleLength);
	if (fread((cSourceString)+szPreambleLength, szSourceLength, 1, pFileStream) != 1)
	{
		fclose(pFileStream);
		free(cSourceString);
		return 0;
	}
	// close the file and return the total length of the combined (preamble + source) string
		fclose(pFileStream);
	if (szFinalLength != 0)
	{
		*szFinalLength = szSourceLength + szPreambleLength;
	}
	cSourceString[szSourceLength + szPreambleLength] = '\0';
	return cSourceString;
}

int main(int argc, char** argv)
{
	int err;                            // error code returned from api calls

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	// OpenCL device memory for matrices
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	// set seed for rand()
	srand(2014);

	//Allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float)* size_A;
	float* h_A = (float*)malloc(mem_size_A);

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float)* size_B;
	float* h_B = (float*)malloc(mem_size_B);

	//Initialize host memory
	randomMemInit(h_A, size_A);
	randomMemInit(h_B, size_B);

	//Allocate host memory for the result C
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float)* size_C;
	float* h_C = (float*)malloc(mem_size_C);

	printf("Initializing OpenCL device...\n");

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source file
	char *KernelSource;
	char *preamble;
	size_t size;
	long lFileSize;

	KernelSource = oclLoadProgSource("lab4.cl", "", &size);

	program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	//
	kernel = clCreateKernel(program, "matrixMul", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the input and output arrays in device memory for our calculation
	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);

	if (!d_A || !d_B || !d_C)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", WA, HA, WB, HB);

	//Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

	int wA = WA;
	int wC = WC;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = 1024;
	globalWorkSize[1] = 1024;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}

	//Retrieve result from device
	err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//print out the result
	//printf("\n\nMatrix C (Results)\n");
	//int i;
	//for(i = 0; i < size_C; i++)
	//{
	//printf("%f ", h_C[i]);
	//if(((i + 1) % WC) == 0)
	//printf("\n");
	//}
	//printf("\n");

	printf("Matrix multiplication completed...\n");
	system("PAUSE");
	//Shutdown and cleanup
	free(h_A);
	free(h_B);
	free(h_C);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_C);
	clReleaseMemObject(d_B);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
