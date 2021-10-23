#include <stdio.h>
#include <windows.h>
#include <stdbool.h>

#define BOARD_SIZE 4
#define NUM_OF_THREADS 8

typedef struct node {
    int* val;
    struct node * next;
} list;

int* create_board();
int get_cores_num();
bool check_position(int* board);
list* find_queens_solutions(int static_queen_row);
void swap(int* array,int e1,int e2);
int* copy_array(int* src, int length);
void print_list(list* head);


int main()
{	
	list* solutions;
	for(int i = 0; i < BOARD_SIZE/2; i++){
		solutions = find_queens_solutions(i);
		print_list(solutions);
		free(solutions);
	}
}

int* create_board() {
	int* board = malloc(sizeof (int) * BOARD_SIZE);
	for(int i=0; i < BOARD_SIZE; i++){
		board[i] = i;
	}
	return board;
}

int get_cores_num() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

bool check_position(int* board) {
	/* Check if any queen is attacking others */
	return true;
}

list* find_queens_solutions(int static_queen_row) {
	/* Generate permutations using heap's algorithm */
	int arr_len = BOARD_SIZE - 1;
	list* current = (list *) malloc(sizeof(list));
	list* head_holder = current;
	int* indexes = malloc(sizeof(int) * arr_len);
	int* position = malloc(sizeof(int) * arr_len + 1);
	for (int i = 0; i < arr_len; i++) {
	    indexes[i] = 0;
		position[i] = i+1 ;
	}
	position[arr_len] = 0;
	current->next = NULL;
	int i = 0;
	while (i < arr_len) {
	    if (indexes[i] < i) {
	        swap(position, i % 2 == 0 ? 0 : indexes[i], i);
			swap(position, static_queen_row, arr_len);
			if(check_position(position)){
				current->next = (list *) malloc(sizeof(list));
				current = current->next;
				current->val = copy_array(position, arr_len + 1);
				current->next = NULL;
			}
			swap(position, static_queen_row, arr_len);
	        indexes[i]++;
	        i = 0;
	    }
	    else {
	        indexes[i] = 0;
	        i++;
	    }
	}
	free(position);
	free(indexes);
	current = head_holder->next;
	free(head_holder);
	return current;
}

void swap(int* array,int e1,int e2) {
	int buff = array[e2];
	array[e2] = array[e1];
	array[e1] = buff;
}

int* copy_array(int* src, int length) {
	int* arr = malloc(sizeof(int) * length);
	for(int i = 0;i < length;i++)
		arr[i]=src[i];
	return arr;
}

void print_list(list* head) {
    list* current = head;

    while (current != NULL) {
		for(int i = 0; i < BOARD_SIZE; i++)
			printf("%d", current->val[i]);
		printf("\n");
        current = current->next;
    }
}