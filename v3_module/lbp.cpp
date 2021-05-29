#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int LookUpTable[65540];
bool check[65540];

// 숫자와 비트수를 인수로 받아서 몇번 비트가 바뀌는지 리턴하는 함수
int getBitChange(int number, int bit_n) {
	int count = 0;
	// 첫번째 비트
	int now = (number&1);		
	//printf("number : %d, bit : %d, i : %d, now : %d\n", number, bit_n, 0, (number >> 0) & 1);
	for (int i = 1; i < bit_n; i++) {
		//printf("number : %d, bit : %d, i : %d, now : %d\n", number, bit_n, i, (number >> i) & 1);
		if (now != ((number >> i) & 1)) {
			count++;
			now = ((number >> i) & 1);
		}
		//printf("count : %d\n", count);
	}
	return count;
}

// arr[]와 bit수 n을 받아서 arr[]에 
// 비트가 2번 이하로 바뀌는 숫자는 count를 1씩 증가시키면서 값을 담고
// 나머지 배열은 마지막 count로 모두 담는 함수.
void getLUT(int arr[], int bit_n) {
	int max = 1;
	int count = 0;
	for (int i = 0; i < bit_n; i++) max *= 2;

	for (int i = 0; i < max; i++) {
		int num = getBitChange(i, bit_n);
		if (num <= 2) {
			LookUpTable[i] = count++;
			check[i] = true;
		}
	}
	for (int i = 0; i < max; i++) {
		if (check[i] == false) LookUpTable[i] = count;
	}
}

int main() {
	int max = 1;
	int bit_n = 16;
	int sum = 0;
	for (int i = 0; i < bit_n; i++) max *= 2;
	getLUT(LookUpTable, bit_n);

	for (int i = 0; i < max; i++) {
		printf("%d\t", LookUpTable[i]);
		sum++;
	}
	printf("\n%d\n", sum);

	return 0;
}