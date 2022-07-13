// Implementation based on the paper "Detecting Single Event Upsets in Embedded Software"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define numThreads 10 // no. of threads will be adjusted based on processor capability and available RAM to maximise detection
int result[numThreads];

void *factorial_10(void* idp){
    result[(int)idp] = 10*9*8*7*6*4*3*2;
    pthread_exit(NULL);
}

void detect_seu(){

    pthread_t factThread[numThreads];
    int ids[numThreads];

    for(int tCnt = 0; tCnt < numThreads; tCnt++){
        ids[tCnt] = tCnt;
        pthread_create(&factThread[tCnt], NULL, factorial_10, &ids[tCnt]);
    }

    for(int tCnt = 0; tCnt < numThreads; tCnt++){
        if(result[tCnt]!=3628800){
            printf("SEU detected!");
            // write into csv file
        }
    }
}