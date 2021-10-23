#include <stdio.h>
#include <windows.h>
#include <stdbool.h>

#define BOARD_SIZE 8
#define NUM_OF_THREADS 8

int* create_board();
int get_cores_num();
bool check_position();

int main()
{			
    printf("%d", get_cores_num());
}

int* create_board() {
	int *board = malloc(sizeof (int) * BOARD_SIZE);
	for(int i=0; i<BOARD_SIZE; i++){
		board[i] = i;
	}
	return board;
}

int get_cores_num() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

bool check_position() {
	/* Implementacja funkcji sprawdzającej czy na szachownicy nic sie nie atakuje */
	return true;
}
