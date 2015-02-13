/*
 Copyright (c) 2015, Burak Sarac, burak@linux.com
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
 following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 following disclaimer in the documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NeuralNetwork.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <pthread.h>
#include <sys/time.h>
using namespace std;

#define E (2.7182818284590452353602874713526624977572470937L )

NeuralNetwork::NeuralNetwork(int noThreads, double* alist, double* blist, int lCount, int* nCounts, int nOfLabels, int yWeight, int xColumnSize) {
	numberOfThreads = noThreads;
	xList = alist;
	yList = blist;
	xColumns = xColumnSize;
	layerCount = lCount;
	neuronCounts = nCounts;
	numberOfLabels = nOfLabels;
	ySize = yWeight;
	ySizeDouble = ySize;
	dMatrixDimensions = new int*[layerCount - 1];
	dLayerCache = new int[layerCount];
	nLayerCache = new int[layerCount + 1];
	eLayerCache = new int[layerCount];
	dMatrixSize = 0;
	xDim2 = neuronCounts[0];
	yDim2 = numberOfLabels;
	nLayerCache[0] = 0;
	eLayerCache[0] = 0;
	dLayerCache[0] = 0;
	deltaSize = 0;
	neuronSize = 0;
	errorSize = 0;
	for (int i = 0; i < layerCount; ++i) {

		neuronSize += i == layerCount - 1 ? neuronCounts[i] : neuronCounts[i] + 1;
		nLayerCache[i + 1] = neuronSize;

		if (i < layerCount - 1) {

			errorSize += i == layerCount - 2 ? neuronCounts[i + 1] : neuronCounts[i + 1] + 1;
			eLayerCache[i + 1] = errorSize;
			dMatrixDimensions[i] = new int[2];
			dMatrixDimensions[i][0] = neuronCounts[i + 1];
			dMatrixDimensions[i][1] = neuronCounts[i] + 1;

			deltaSize += (dMatrixDimensions[i][0] * dMatrixDimensions[i][1]);
			dLayerCache[i + 1] = deltaSize;
		}
	}

	mDeltaSize = sizeof(double) * deltaSize;


}

NeuralNetwork::~NeuralNetwork() {
	delete[] dLayerCache;
	delete[] nLayerCache;
	delete[] eLayerCache;
	free(xList);
	delete[] yList;

}

void* NeuralNetwork::calculateBackCost(void *dat) {
	struct stData* data = (struct stData*) dat;
	data->cost = 0;
	double* neurons = (double*) malloc(sizeof(double) * data->neuronSize);
	double* errors = (double*) malloc(sizeof(double) * data->errorSize);
	data->deltas = (double*) malloc(sizeof(double) * data->deltaSize);
	for (int i = 0; i < data->deltaSize; ++i) {
		data->deltas[i] = 0;
	}
	int layerCount = data->layerCount;
	int* neuronCounts = data->neuronCounts;
	double* thetas = data->thetas;
	double* xList = data->xList;
	double* yList = data->yList;
	int* dlayerCache = data->dlayerCache;
	int numLabels = data->numLabels;
	int xListRows = data->xListRows;
	int** dMatrixInfo = data->dMatrixInfo;
	int* nLayerCache = data->nLayerCache;
	int* eLayerCache = data->eLayerCache;

	for (int m = data->loopMin; m < data->loopMax; m++) {
		int yCache = m * numLabels;
		int xCache = xListRows * m;
		double* x = &(xList[xCache]);
		double* y = &(yList[yCache]);

		//forward propagate
		for (int l = 0; l < layerCount; l++) {
			int previousLayer = nLayerCache[l];

			int neuronSize = l == layerCount - 1 ? neuronCounts[l] : neuronCounts[l] + 1;
			for (int j = 0; j < neuronSize; j++) {
				int row = previousLayer + j;
				neurons[row] = 0;

				if (j == 0 && l != layerCount - 1) {
					neurons[row] = 1;
				} else if (l == 0) {
					neurons[row] = x[(j - 1)];
				} else {
					int dCache = dlayerCache[l - 1];
					int pNCache = nLayerCache[l - 1];
					int index = l == layerCount - 1 ? j : j - 1;
					int dRowCache = (dMatrixInfo[l - 1][1] * index) + dCache;
					int nCounts = neuronCounts[l - 1] + 1;
					double* t = &(thetas[dRowCache]);
					double* n = &(neurons[pNCache]);
					for (int k = 0; k < nCounts; k++) {
						neurons[row] += t[k] * n[k];
					}

					neurons[row] = 1 / (1 + pow(E, -1 * neurons[row]));

					if(l == layerCount - 1 && neurons[row] == 1){
						neurons[row] -= 0.00000000001;
					}

				}
			}
		}

		//backpropagate
		for (int i = layerCount - 2; i >= 0; i--) {

			int neuronSize = i == layerCount - 2 ? neuronCounts[i + 1] : neuronCounts[i + 1] + 1;
			int previousLayer = eLayerCache[i];
			int nCache = nLayerCache[i + 1];

			int dCache = dlayerCache[i + 1];
			int eCache = eLayerCache[i + 1];
			double* e = &(errors[eCache]);
			double* t = &(thetas[dCache]);
			for (int j = neuronSize - 1; j >= 0; j--) {
				int row = previousLayer + j;

				errors[row] = 0; //reset
				double nVal = neurons[nCache + j];
				if (i == layerCount - 2) {
					errors[row] = nVal - y[j];
				} else {
					int nCounts = neuronCounts[i + 2];
					int isLast = nCounts - 1;
					double sigmoid = (nVal * (1 - nVal));
					double* t2 = &(t[j]);
					int dif = nCounts % 4;
					int siz = nCounts - dif;
					for (int k = 0; k < siz; k = k + 4) {
						int r = (dMatrixInfo[i + 1][1] * k);
						int r1 = (dMatrixInfo[i + 1][1] * (k + 1));
						int r2 = (dMatrixInfo[i + 1][1] * (k + 2));
						int r3 = (dMatrixInfo[i + 1][1] * (k + 3));
						errors[row] += t2[r] * e[k];
						errors[row] += t2[r1] * e[k + 1];
						errors[row] += t2[r2] * e[k + 2];
						errors[row] += t2[r3] * e[k + 3];

					}

					for (int a = 0; a < dif; a++) {
						int k = siz + a;
						int r = (dMatrixInfo[i + 1][1] * k);
						errors[row] += t2[r] * e[k];

						if (k == isLast) {
							errors[row] = errors[row] * sigmoid;
						}
					}
				}

			}
		}

		//calculate deltas
		double sum = 0.0;
		for (int i = 0; i < layerCount - 1; i++) {
			int n1 = neuronCounts[i + 1];
			int n2 = neuronCounts[i] + 1;
			int nCache1 = nLayerCache[i + 1];
			int eCache = eLayerCache[i];
			int nCache = nLayerCache[i];
			double* e = &(errors[eCache]);
			double* n = &(neurons[nCache]);
			int isLast = i == layerCount - 2;
			int dCache = dlayerCache[i];
			double* d = &(data->deltas[dCache]);
			int dif = n1 % 4;
			int siz = n1 - dif;
			for (int j = 0; j < siz; j = j + 4) {
				if (isLast) {

					sum += ((-1 * yList[yCache + j]) * log(neurons[nCache1 + j])) - ((1 - yList[yCache + j]) * log(1 - neurons[nCache1 + j]));
					sum += ((-1 * yList[yCache + j + 1]) * log(neurons[nCache1 + j + 1])) - ((1 - yList[yCache + j + 1]) * log(1 - neurons[nCache1 + j + 1]));
					sum += ((-1 * yList[yCache + j + 2]) * log(neurons[nCache1 + j + 2])) - ((1 - yList[yCache + j + 2]) * log(1 - neurons[nCache1 + j + 2]));
					sum += ((-1 * yList[yCache + j + 3]) * log(neurons[nCache1 + j + 3])) - ((1 - yList[yCache + j + 3]) * log(1 - neurons[nCache1 + j + 3]));
				}
				int index = i == 0 ? j + 1 : j;
				int index2 = index + 1;
				int index3 = index + 2;
				int index4 = index + 3;
				int drcache = (dMatrixInfo[i][1] * j);
				int drcache2 = (dMatrixInfo[i][1] * (j + 1));
				int drcache3 = (dMatrixInfo[i][1] * (j + 2));
				int drcache4 = (dMatrixInfo[i][1] * (j + 3));
				double eVal = e[index];
				double eVal2 = e[index2];
				double eVal3 = e[index3];
				double eVal4 = e[index4];
				double* d2 = &(d[drcache]);
				double* d22 = &(d[drcache2]);
				double* d23 = &(d[drcache3]);
				double* d24 = &(d[drcache4]);
				int diff = n2 % 4;
				int size = n2 - diff;
				for (int k = 0; k < size; k = k + 4) {
					d2[k] += eVal * n[k];
					d2[k + 1] += eVal * n[k + 1];
					d2[k + 2] += eVal * n[k + 2];
					d2[k + 3] += eVal * n[k + 3];

					d22[k] += eVal2 * n[k];
					d22[k + 1] += eVal2 * n[k + 1];
					d22[k + 2] += eVal2 * n[k + 2];
					d22[k + 3] += eVal2 * n[k + 3];

					d23[k] += eVal3 * n[k];
					d23[k + 1] += eVal3 * n[k + 1];
					d23[k + 2] += eVal3 * n[k + 2];
					d23[k + 3] += eVal3 * n[k + 3];

					d24[k] += eVal4 * n[k];
					d24[k + 1] += eVal4 * n[k + 1];
					d24[k + 2] += eVal4 * n[k + 2];
					d24[k + 3] += eVal4 * n[k + 3];
				}
				for (int d = 0; d < diff; d++) {
					double nVal = n[size + d];
					d2[size + d] += eVal * nVal;
					d22[size + d] += eVal2 * nVal;
					d23[size + d] += eVal3 * nVal;
					d24[size + d] += eVal4 * nVal;
				}
			}

			for (int a = 0; a < dif; a++) {
				int j = a + siz;
				if (isLast) {

					sum += ((-1 * yList[yCache + j]) * log(neurons[nCache1 + j])) - ((1 - yList[yCache + j]) * log(1 - neurons[nCache1 + j]));
				}
				int index = i == 0 ? j + 1 : j;
				int drcache = (dMatrixInfo[i][1] * j);
				double eVal = e[index];
				double* d2 = &(d[drcache]);
				int diff = n2 % 4;
				int size = n2 - diff;
				for (int k = 0; k < size; k = k + 4) {
					d2[k] += eVal * n[k];
					d2[k + 1] += eVal * n[k + 1];
					d2[k + 2] += eVal * n[k + 2];
					d2[k + 3] += eVal * n[k + 3];
				}
				for (int d = 0; d < diff; d++) {
					double nVal = n[size + d];
					d2[size + d] += eVal * nVal;
				}
			}

		}
		double ySizeDouble = data->ySize;
		data->cost += (1 / ySizeDouble) * sum;

	}

	free(neurons);
	free(errors);

	if (!data->isLast) {
		pthread_exit(NULL);
	}

	return 0;
}

GradientParameter* NeuralNetwork::calculateBackCostWithThetas(double lambda, double* thetas) {
	//allocate place for deltas
	deltas = (double *) malloc(mDeltaSize);

	//create threads according to params
	pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * (numberOfThreads - 1));
	//we need rowcount in double value for calculation
	double ySizeDouble = ySize;
	//create params for each thread
	struct stData* stDatas = (struct stData*) malloc(sizeof(struct stData) * numberOfThreads);
	double cost = 0;
	double** pDeltas = (double**) malloc(sizeof(double*) * numberOfThreads);

	for (int t = 0; t < numberOfThreads; ++t) {

		int isLast = t == (numberOfThreads - 1);
		int loopMin = (int) ((long) (t + 0) * (long) (ySize) / (long) numberOfThreads);
		int loopMax = (int) ((long) (t + 1) * (long) (ySize) / (long) numberOfThreads);

		stDatas[t].deltas = &(pDeltas[t][0]);
		stDatas[t].xList = xList;
		stDatas[t].ySize = ySize;
		stDatas[t].yList = yList;
		stDatas[t].layerCount = layerCount;
		stDatas[t].neuronCounts = neuronCounts;
		stDatas[t].lambda = lambda;
		stDatas[t].thetas = thetas;
		stDatas[t].neuronSize = neuronSize;
		stDatas[t].errorSize = errorSize;
		stDatas[t].deltaSize = deltaSize;
		stDatas[t].xListRows = xColumns;
		stDatas[t].dlayerCache = dLayerCache;
		stDatas[t].dMatrixInfo = dMatrixDimensions;
		stDatas[t].nLayerCache = nLayerCache;
		stDatas[t].eLayerCache = eLayerCache;
		stDatas[t].numLabels = numberOfLabels;
		stDatas[t].cost = 0;
		stDatas[t].isLast = isLast;
		stDatas[t].loopMin = loopMin;
		stDatas[t].loopMax = loopMax;

		if (!isLast) {
			pthread_create(&threads[t], NULL, calculateBackCost, (void *) &(stDatas[t]));

		} else {
			//if its last handle by main thread
			this->calculateBackCost(&stDatas[t]);
		}

	}


	//wait for other threads
	for (int t = 0; t < numberOfThreads - 1; t++) {
		pthread_join(threads[t], NULL);
	}

	double thetaSum = 0.0;

	//collect all data from threads and update cost
	int da = 0;
	for (int l = 0; l < deltaSize; l++) {
		int dc = (l - dLayerCache[da]) % dMatrixDimensions[da][1];
		deltas[l] = 0;
		for (int i = 0; i < numberOfThreads; i++) {

			deltas[l] += stDatas[i].deltas[l];

		}
		deltas[l] = (1 / ySizeDouble) * deltas[l];
		if (dc > 0) {
			deltas[l] += (lambda / ySizeDouble) * thetas[l];
			thetaSum += pow(thetas[l], 2);
		}

		if ((l + 1) == dLayerCache[da + 1]) {
			da++;
		}
	}

	for (int i = 0; i < numberOfThreads; ++i) {
		cost += stDatas[i].cost;
		free(stDatas[i].deltas);
	}

	thetaSum = ((lambda / (2 * ySizeDouble)) * thetaSum);

	cost += thetaSum;

	free(threads);
	free(stDatas);
	free(pDeltas);
	return new GradientParameter(deltas, cost);;

}

void NeuralNetwork::predict(double* tList, double* yTemp) {

	int totalCorrect = 0;
	int totalWrong = 0;

	for (int i = 0; i < ySize; ++i) {

		double* neurons = forwardPropogate(i, tList, &(xList[(i * xColumns)]));
		double closer = RAND_MAX;
		double val = 0;
		for (int j = 0; j < numberOfLabels; j++) {

			if (fabs((1 - closer)) > fabs((1 - neurons[nLayerCache[layerCount - 1] + j]))) {
				val = j + 1;
				closer = neurons[nLayerCache[layerCount - 1] + j];
			}
		}

		if (yTemp[i] == val) {
			totalCorrect++;
		} else {
			totalWrong++;
		}

		free(neurons);

	}

	printf("\nPrediction complete. Total %i correct and %i wrong prediction\n", totalCorrect, totalWrong);
	double successRate = totalCorrect * 100 / ySize;
	printf("\n Success rate is: %%%0.0f", successRate);
}
double* NeuralNetwork::forwardPropogate(int aListIndex, double* tList, double* xList) {

	int mNeuronSize = sizeof(double) * neuronSize;
	double* neurons = (double *) malloc(mNeuronSize);
	for (int l = 0; l < layerCount; l++) {
		int previousLayer = nLayerCache[l];

		int neuronSize = l == layerCount - 1 ? neuronCounts[l] : neuronCounts[l] + 1;
		for (int j = 0; j < neuronSize; j++) {
			int row = previousLayer + j;
			neurons[row] = 0;

			if (j == 0 && l != layerCount - 1) {
				neurons[row] = 1;
			} else if (l == 0) {
				neurons[row] = xList[(j - 1)];
			} else {
				int dCache = dLayerCache[l - 1];
				int pNCache = nLayerCache[l - 1];
				int index = l == layerCount - 1 ? j : j - 1;
				int dRowCache = (dMatrixDimensions[l - 1][1] * index) + dCache;
				int nCounts = neuronCounts[l - 1] + 1;
				double* t = &(tList[dRowCache]);
				double* n = &(neurons[pNCache]);
				for (int k = 0; k < nCounts; k++) {
					neurons[row] += t[k] * n[k];
				}

				neurons[row] = 1 / (1 + pow(E, -1 * neurons[row]));

			}
		}
	}

	return neurons;
}

