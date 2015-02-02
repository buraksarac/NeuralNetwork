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

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_
#include "GradientParameter.h"
struct stData {
	double* deltas;
	double* xList;
	int ySize;
	double* yList;
	int layerCount;
	int* neuronCounts;
	double lambda;
	double* thetas;
	int neuronSize;
	int errorSize;
	int deltaSize;
	int xListRows;
	int* dlayerCache;
	int** dMatrixInfo;
	int* nLayerCache;
	int* eLayerCache;
	int numLabels;
	double cost;
	int isLast;
	int loopMin;
	int loopMax;
};
class NeuralNetwork {
private:
	int layerCount;
	int* neuronCounts;
	int numberOfLabels;
	int ySize;
	double ySizeDouble;
	int** dMatrixDimensions;
	int* dLayerCache;
	int* nLayerCache;
	int* eLayerCache;
	int dMatrixSize;
	int xDim2;
	int yDim2;
	int neuronSize;
	int errorSize;
	int deltaSize;
	int mDeltaSize;
	double* deltas;
	int xColumns;
	double* xList;
	double* yList;
	int numberOfThreads;
public:
	NeuralNetwork(int noThreads, double* alist, double* blist, int layerCount, int* neuronCounts, int numberOfLabels, int ySize, int xColumnSize);
	GradientParameter* calculateBackCostWithThetas(double lambda, double* thetas);
	static void* calculateBackCost(void *dat);
	double* forwardPropogate(int aListIndex, double* tList, double* xList);
	void predict(double* tList, double* yTemp);
	virtual ~NeuralNetwork();
};

#endif /* SRC_NEURALNETWORK_H_ */
