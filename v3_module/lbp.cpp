#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int LookUpTable[65540];
bool check[65540];

// ���ڿ� ��Ʈ���� �μ��� �޾Ƽ� ��� ��Ʈ�� �ٲ���� �����ϴ� �Լ�
int getBitChange(int number, int bit_n) {
	int count = 0;
	// ù��° ��Ʈ
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

// arr[]�� bit�� n�� �޾Ƽ� arr[]�� 
// ��Ʈ�� 2�� ���Ϸ� �ٲ�� ���ڴ� count�� 1�� ������Ű�鼭 ���� ���
// ������ �迭�� ������ count�� ��� ��� �Լ�.
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