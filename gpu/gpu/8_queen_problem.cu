#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <stdbool.h>
#include "device_launch_parameters.h"

#define BOARD_SIZE 12
#define BLOCKS_PER_GRID 128
#define THREADS_PER_BLOCK 1024
#define MAX_SOLUTIONS 1000000

typedef struct node {
	int* val;
	struct node * next;
} list;

/* Generate board with one queen each row */
int* generate_base_board();

/* It is used for devide permutations */
int get_static_rows();

/* Returns base permutations that are used for generate rest ones */
list* generate_boards(int* board_base, int static_rows, int pos);

/* Check if the position has non attacking queens */
bool check_position(int* board);

/* Helpful methods */
void swap(int* array, int e1, int e2);
int* copy_array(int* src, int length);
int* copy_array_gpu(int* src, int length);
void merge_lists(list* list1, list* list2);
int size_of(list* head);
void DisplayHeader();
void list_to_array(list* boards, int* boards_array);

/* Threads */
list* boards;
int static_rows = 0;
int solutions_count = 0;

__device__ void gpu_swap(int* array, int e1, int e2) {
	int buff = array[e2];
	array[e2] = array[e1];
	array[e1] = buff;
}

__device__ bool gpu_check_position(int* board) {
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = i + 1; j < BOARD_SIZE; j++)
		{
			bool onTheSameDiagonal = ((board[i] - i == board[j] - j) || (board[i] + i == board[j] + j));
			if (onTheSameDiagonal) return false;
		}
	}
	return true;
}

__global__ void find_queens_solutions(int* boards, int static_rows, int* solutions, int* mutex, int* solutionIdx) {

	int threadId = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int num_of_boards = 1;
	for (int i = BOARD_SIZE; i > BOARD_SIZE - static_rows; i--) {
		num_of_boards *= i;
	}
	int board[BOARD_SIZE];
	int indexes[BOARD_SIZE];
	while (threadId < num_of_boards) {
		int boardIdx = threadId * BOARD_SIZE;
		for (int i = 0; i < BOARD_SIZE; i++) {
			board[i] = boards[i + boardIdx];
			indexes[i] = 0;
		}

		/* Generate permutations using heap's algorithm */
		int i = 0;
		while (i < BOARD_SIZE - static_rows) {
			if (indexes[i] < i) {
				gpu_swap(board, i % 2 == 0 ? 0 : indexes[i], i);
				if (gpu_check_position(board)) {
					

				}
				indexes[i]++;
				i = 0;
			}
			else {
				indexes[i] = 0;
				i++;
			}
		}
		threadId += BLOCKS_PER_GRID * THREADS_PER_BLOCK;
	}
}

int main() {
	DisplayHeader();
	clock_t t = clock();
	static_rows = get_static_rows();
	int* base_board = generate_base_board();
	boards = generate_boards(base_board, static_rows, 0);

	int* boards_array = (int*)malloc(size_of(boards) * BOARD_SIZE * sizeof(int));
	list_to_array(boards, boards_array);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int *boards_on_gpu = 0;
	cudaMalloc((void**)&boards_on_gpu, size_of(boards) * BOARD_SIZE * sizeof(int));
	cudaMemcpy(boards_on_gpu, boards_array, size_of(boards) * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	int* solutions = 0;
	cudaMalloc((void**)&solutions, MAX_SOLUTIONS * BOARD_SIZE * sizeof(int));

	int *mutex;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));
	int *solutionIdx;
	cudaMalloc((void **)&solutionIdx, sizeof(int));
	cudaMemset(solutionIdx, 0, sizeof(int));
	find_queens_solutions <<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>> (boards_on_gpu, static_rows, solutions, mutex, solutionIdx);

	int* number_of_solutions = (int*)malloc(sizeof(int));
	cudaMemcpy(number_of_solutions, solutionIdx, sizeof(int), cudaMemcpyDeviceToHost);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching find_queens_solutions!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; 
	printf("The program took %f seconds to execute\n", time_taken);
	printf("%s%d", "Number of solutions: ", *number_of_solutions);

Error:
	cudaFree(boards_on_gpu);
	cudaFree(solutions);
	cudaFree(mutex);
	cudaFree(solutionIdx);
	return 0;
}

void list_to_array(list* boards, int* boards_array) {
	list* current = boards;
	int j = 0;
	while (current != NULL) {
		for (int i = 0; i < BOARD_SIZE; i++)
			boards_array[j++] = current->val[i];

		current = current->next;
	}
}

int* generate_base_board() {
	int* base_board = (int*)malloc(sizeof(int) * BOARD_SIZE);
	for (int i = 0; i < BOARD_SIZE; i++)
		base_board[i] = i;
	return base_board;
}

int get_static_rows() {
	int n = BOARD_SIZE;
	int factorial = 1;
	while (factorial < BLOCKS_PER_GRID * THREADS_PER_BLOCK) {
		factorial *= n--;
	}
	return BOARD_SIZE - n;
}

list* generate_boards(int* board_base, int static_rows, int pos) {
	list* boards = NULL;
	if (pos == static_rows) {
		boards = (list*)malloc(sizeof(list));
		boards->val = copy_array(board_base, BOARD_SIZE);
		boards->next = NULL;
		return boards;
	}
	for (int k = BOARD_SIZE - pos - 1; k >= 0; k--) {
		swap(board_base, BOARD_SIZE - pos - 1, k);
		list* temp = generate_boards(board_base, static_rows, pos + 1);
		merge_lists(temp, boards);
		boards = temp;
		swap(board_base, BOARD_SIZE - pos - 1, k);
	}
	return boards;
}

void swap(int* array, int e1, int e2) {
	int buff = array[e2];
	array[e2] = array[e1];
	array[e1] = buff;
}

void merge_lists(list* list1, list* list2) {
	list* current = list1;
	while (current->next != NULL) {
		current = current->next;
	}
	current->next = list2;
}

int* copy_array(int* src, int length) {
	int* arr = (int*)malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
		arr[i] = src[i];
	return arr;
}

void DisplayHeader()
{
	const int kb = 1024;
	const int mb = kb * kb;
	fprintf(stderr, "NBody.GPU\n\n");

	fprintf(stderr, "CUDA version:   v%d\n", CUDART_VERSION);

	int devCount;
	cudaGetDeviceCount(&devCount);
	fprintf(stderr, "CUDA Devices\n\n");

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		fprintf(stderr, "%d:%s:%d:%d \n", i, props.name, props.major, props.minor);
		fprintf(stderr, "Global memory:    %d mb \n", props.totalGlobalMem / mb);
		fprintf(stderr, "Shared memory:    %d kb \n", props.sharedMemPerBlock / kb);
		fprintf(stderr, "Constant memory:    %d kb \n", props.totalConstMem / kb);
		fprintf(stderr, "Block registers:    %d  \n", props.regsPerBlock);
		fprintf(stderr, "Warp size:    %d  \n", props.warpSize);
		fprintf(stderr, "Threads per block:    %d  \n", props.maxThreadsPerBlock);
		fprintf(stderr, "Max block dimensions: [ %d, %d, %d ] \n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		fprintf(stderr, "Max grid dimensions: [ %d, %d, %d ] \n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

	}
}





int size_of(list* head) {
	list* current = head;
	int size = 0;
	while (current != NULL) {
		size++;
		current = current->next;
	}
	return size;
}