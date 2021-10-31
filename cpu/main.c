#include <stdio.h>
#include <windows.h>
#include <stdbool.h>
#include <pthread.h>

#define BOARD_SIZE 8
#define NUM_OF_THREADS 4

typedef struct node {
    int* val;
    struct node * next;
} list;

/* Generate board with one queen each row */
int* generate_base_board();

/* It is used for devide permutations */
int get_static_rows();

/* Returns base permutations that are used for generate rest ones */
list* generate_boards(int* board_base ,int static_rows, int pos);

/* Returns non-attacking boards */
list* find_queens_solutions(int* board, int static_rows);

/* Check if the position has non attacking queens */
bool check_position(int* board);

/* Helpful methods */
int get_cores_num();
void swap(int* array,int e1,int e2);
int* copy_array(int* src, int length);
void print_list(list* head);
void merge_lists(list* list1, list* list2);
int size_of(list* head);

/* Threads */
pthread_mutex_t lock;
list* boards;
int static_rows = 0;
int solutions_count = 0;
void* start_thread();

int main()
{
	static_rows = get_static_rows();	
	int* base_board = generate_base_board();
	
	/* Each board represents set of permutations */
	boards = generate_boards(base_board, static_rows, 0);
	
	/* Init threads */
	int err = pthread_mutex_init(&lock, NULL);
	
    if (err)
    {
        printf("\nMutex init error!");
        return 1;
    }
	
	pthread_t tid[NUM_OF_THREADS];
	
	for(int i = 0; i < NUM_OF_THREADS; i++){
		err = pthread_create(&(tid[i]), NULL, &start_thread, NULL);
		if(err){
			printf("\nThread init error!");
			return 1;
		}
	}
	
	/* Clear threads */
	for(int i = 0; i < NUM_OF_THREADS; i++){
		pthread_join(tid[i], NULL);
	}
	
    pthread_mutex_destroy(&lock);
	
	/* Print results */
	printf("%s%d","Number of solutions: ", solutions_count);
	
	return 0;
}

int* generate_base_board(){
	int* base_board =  malloc(sizeof(int) * BOARD_SIZE);
	for(int i=0;i<BOARD_SIZE;i++)
		base_board[i]=i;
	return base_board;
}

int get_static_rows(){
	int n = BOARD_SIZE;
	int factorial = 1;
	while(factorial < NUM_OF_THREADS) {
		factorial *= n--;
	}
	return BOARD_SIZE - n;
}

list* generate_boards(int* board_base ,int static_rows, int pos){
	list* boards = NULL;
	if(pos == static_rows){
		boards = (list*) malloc(sizeof(list));
		boards->val = copy_array(board_base, BOARD_SIZE);
		boards->next = NULL;
		return boards;
	}
	for(int k = BOARD_SIZE - pos - 1;k >= 0; k--){
		swap(board_base, BOARD_SIZE - pos - 1, k);
		list* temp = generate_boards(board_base, static_rows, pos+1);
		merge_lists(temp, boards);
		boards = temp;
		swap(board_base, BOARD_SIZE - pos - 1, k);
	}
	return boards;
}

list* find_queens_solutions(int* board, int static_rows) {
	list* current = (list*) malloc(sizeof(list));
	list* head_holder = current;
	int* indexes = malloc(sizeof(int) * BOARD_SIZE);
	for (int i = 0; i < BOARD_SIZE; i++) {
	    indexes[i] = 0;
	}
	current->next = NULL;
	
	/* Generate permutations using heap's algorithm */
	int i = 0;
	while (i < BOARD_SIZE - static_rows) {
	    if (indexes[i] < i) {
	        swap(board, i % 2 == 0 ? 0 : indexes[i], i);
			if(check_position(board)){
				current->next = (list *) malloc(sizeof(list));
				current = current->next;
				current->val = copy_array(board, BOARD_SIZE);
				current->next = NULL;
			}
	        indexes[i]++;
	        i = 0;
	    }
	    else {
	        indexes[i] = 0;
	        i++;
	    }
	}
	free(indexes);
	current = head_holder->next;
	free(head_holder);
	return current;
}

bool check_position(int* board) {
	for(int i=0; i<BOARD_SIZE; i++){
		for (int j = i+1; j < BOARD_SIZE; j++)
		{
			bool onTheSameDiagonal =  ((board[i]-i == board[j]-j ) || (board[i]+i == board[j]+j)) ;
			if(onTheSameDiagonal) return false;
		}
	}
	return true;
}

int get_cores_num() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
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

void merge_lists(list* list1, list* list2){
	list* current = list1;
    while (current->next != NULL) {
        current = current->next;
    }
	current->next = list2;
}

int size_of(list* head){
	list* current = head;
	int size = 0;
    while (current != NULL) {
		size++;
        current = current->next;
    }
	return size;
}

void* start_thread(){
	list* solutions = NULL;
	while(1){
		pthread_mutex_lock(&lock);
		
		solutions_count += size_of(solutions);
		
		if(boards == NULL){
			pthread_mutex_unlock(&lock);
			return NULL;
		}
		
		int* position = boards->val;
		boards = boards->next;
		
		pthread_mutex_unlock(&lock);
		
		solutions = find_queens_solutions(position, static_rows);
		/* If you want to print solutions */
		/* print_list(solutions); */
	}
}