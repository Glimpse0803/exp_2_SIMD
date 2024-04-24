#include<iostream>
#include<xmmintrin.h> //SSE
#include<emmintrin.h> //SSE2
#include<pmmintrin.h> //SSE3
#include<tmmintrin.h> //SSSE3
#include<smmintrin.h> //SSE4.1
#include<nmmintrin.h> //SSSE4.2
#include<immintrin.h> //AVX、AVX2
#include<windows.h>
#include<iomanip>
#include<ctime>
using namespace std;
void generateSample(float** A, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			A[i][j] = 0;//下三角赋值为0;
		}
		A[i][i] = 1.0;//对角线赋值为1;
		for (int j = i; j < N; j++) {
			A[i][j] = rand();//上三角赋值为任意值;
		}
	}
	for (int k = 0; k < N; k++) {
		for (int i = k + 1; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] += A[k][j];//每一行都加上比自己下标小的行;
			}
		}
	}
}
void show(float** A, int N) {//打印结果;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << fixed << setprecision(0) << A[i][j] << " ";
		}
		cout << endl;
	}
}

//不对齐算法:
void misalignSolution(float** A, int N) {
	for (int k = 0; k < N; k++) {
		__m256 vt = _mm256_set1_ps(A[k][k]);
		for (int j = k + 1; j + 8 <= N; j += 8) {
			//开头强制不对齐:
			if (j == k + 1) {//第一轮打包;
				unsigned long addr = (unsigned long)(&A[k][j]);//获取第一个元素的地址;
				int offset = 8 - (addr % 32) / 8;//偏移量+1，强制不对齐;
				for (int p = 0; p < offset; p++) {
					A[k][j] /= A[k][k];
					j++;
				}
				continue;
			}
			__m256 va = _mm256_load_ps(&A[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&A[k][j], va);
			if (j + 16 > N) {//处理末尾
				while (j < N) {
					A[k][j] /= A[k][k];
					j++;
				}
				break;
			}
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			__m256 vaik = _mm256_set1_ps(A[i][k]);
			for (int j = k + 1; j + 8 <= N; j += 8) {
				//开头强制不对齐:
				if (j == k + 1) {//第一轮打包;
					unsigned long addr = (unsigned long)(&A[k][j]);//获取第一个元素的地址;
					int offset = 8 - (addr % 32) / 8 + 1;//偏移量+1，强制不对齐;
					for (int p = 0; p < offset; p++) {
						A[i][j] -= A[i][k] * A[k][j];
						j++;
					}
					continue;
				}
				__m256 vakj = _mm256_load_ps(&A[k][j]);
				__m256 vaij = _mm256_load_ps(&A[i][j]);
				__m256 vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&A[i][j], vaij);
				if (j + 16 > N) {//处理末尾
					while (j < N) {
						A[i][j] -= A[i][k] * A[k][j];
						j++;
					}
					break;
				}
			}
			A[i][k] = 0;
		}
	}
}

//对齐算法:
void alignSolution(float** A, int N) {
	for (int k = 0; k < N; k++) {
		__m256 vt = _mm256_set1_ps(A[k][k]);
		for (int j = k + 1; j + 8 <= N; j += 8) {
			//开头对齐:
			if (j == k + 1) {//第一轮打包;
				unsigned long addr = (unsigned long)(&A[k][j]);//获取第一个元素的地址;
				int offset = 8 - (addr % 32) / 8;//计算偏移量;
				for (int p = 0; p < offset; p++) {
					A[k][j] /= A[k][k];
					j++;
				}
				continue;
			}
			__m256 va = _mm256_load_ps(&A[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&A[k][j], va);
			if (j + 16 > N) {//处理末尾
				while (j < N) {
					A[k][j] /= A[k][k];
					j++;
				}
				break;
			}
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			__m256 vaik = _mm256_set1_ps(A[i][k]);
			for (int j = k + 1; j + 8 <= N; j += 8) {
				//开头对齐:
				if (j == k + 1) {//第一轮打包;
					unsigned long addr = (unsigned long)(&A[k][j]);//获取第一个元素的地址;
					int offset = 8 - (addr % 32) / 8;//计算偏移量;
					for (int p = 0; p < offset; p++) {
						A[i][j] -= A[i][k] * A[k][j];
						j++;
					}
					continue;
				}
				__m256 vakj = _mm256_load_ps(&A[k][j]);
				__m256 vaij = _mm256_load_ps(&A[i][j]);
				__m256 vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&A[i][j], vaij);
				if (j + 16 > N) {//处理末尾
					while (j < N) {
						A[i][j] -= A[i][k] * A[k][j];
						j++;
					}
					break;
				}
			}
			A[i][k] = 0;
		}
	}
}

int main() {
	float** A;
	float** B;
	int N = 1280;
	A = new float* [N];
	for (int i = 0; i < N; i++) {
		A[i] = new float[N];//申请空间;
	}
	B = new float* [N];
	for (int i = 0; i < N; i++) {
		B[i] = new float[N];//申请空间;
	}

	long long head_1, tail_1, freq;
	long long head_2, tail_2;
	int step = 64;//确定跨度;
	for (int n = step; n <= N; n += step) {//确定问题的规模

		generateSample(A, N);
		generateSample(B, N);
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

		//串行算法:
		QueryPerformanceCounter((LARGE_INTEGER*)&head_1);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail_1);
		long long limit = 0.1 * freq;
		int counter_1 = 0;
		while ((tail_1 - head_1) < limit) {
			misalignSolution(A, n);
			counter_1++;
			QueryPerformanceCounter((LARGE_INTEGER*)&tail_1);
		}

		//并行算法:
		QueryPerformanceCounter((LARGE_INTEGER*)&head_2);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail_2);
		int counter_2 = 0;
		while ((tail_2 - head_2) < limit) {
			alignSolution(B, n);
			counter_2++;
			QueryPerformanceCounter((LARGE_INTEGER*)&tail_2);
		}

		//时间统计:
		float seconds_1 = (tail_1 - head_1) * 1000.0 / freq;
		float seconds_2 = (tail_2 - head_2) * 1000.0 / freq;
		cout << fixed << setprecision(5);
		cout << "问题规模" << n << "	未对齐算法平均用时:" << seconds_1 / counter_1;
		cout << "	对齐算法平均用时" << seconds_2 / counter_2 << endl;
	}
}
