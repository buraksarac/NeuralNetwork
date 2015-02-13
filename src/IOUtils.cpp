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

#include "IOUtils.h"
#include <iostream>
#include <cmath>
#include <math.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <sys/time.h>
#include <sys/stat.h>
struct timeval timeValue;
IOUtils::IOUtils() {
	// TODO Auto-generated constructor stub

}

int IOUtils::fileExist(string name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);

}
double* IOUtils::getFeaturedList(double* list, int columnSize, int rowSize) {

	double* sums = (double*) malloc(sizeof(double) * rowSize);
	double* means = (double*) malloc(sizeof(double) * rowSize);
	double* stds = (double*) malloc(sizeof(double) * rowSize);
	double* featuredList = (double*) malloc(sizeof(double) * rowSize * columnSize);

	for (int i = 0; i < rowSize; ++i) {
		double sum = 0.0;
		double correction = 0.0;
		for (int j = 0; j < columnSize; ++j) {
			double y = list[(i * columnSize) + j] - correction;
			double t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		sums[i] = sum;
		means[i] = sums[i] / columnSize;
	}

	for (int i = 0; i < rowSize; ++i) {
		double sum = 0.0;
		double correction = 0.0;
		for (int j = 0; j < columnSize; ++j) {
			double value = std::pow((list[(i * columnSize) + j] - means[i]), 2);
			double y = value - correction;
			double t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		stds[i] = sum;
	}

	for (int i = 0; i < rowSize; ++i) {
		stds[i] = sqrt(stds[i] / columnSize);
	}

	for (int i = 0; i < rowSize; ++i) {
		for (int j = 0; j < columnSize; ++j) {
			featuredList[(i * columnSize) + j] = (list[(i * columnSize) + j] - means[j]) / stds[j];
		}
	}

	free(sums);
	free(means);
	free(stds);
	free(list);
	return featuredList;
}
void IOUtils::saveThetas(double* thetas, lint size) {
	gettimeofday(&timeValue, NULL);
	string fileName = "thetas_";
	std::stringstream sstm;
	sstm << fileName << timeValue.tv_sec << ".dat";

	ofstream f(sstm.str().c_str());
	copy(thetas, thetas + size, ostream_iterator<double>(f, "\n"));
	printf("Thetas (%s) has been saved into project folder.", sstm.str().c_str());
}

double* IOUtils::getArray(string path, lint rows, lint columns) {

	ifstream inputStream;

	lint currentRow = 0;
	std::string s;
	inputStream.open(path.c_str());

	if(!inputStream.is_open()){
		throw 3;
	}
	lint size = columns * rows;
	lint mListSize = sizeof(double) * size;
	double* list = (double *) malloc(mListSize);

	while (!inputStream.eof()) {

		if (currentRow < size) {

			inputStream >> s;
			try {
				list[currentRow++] = strtod(s.c_str(), NULL);
			} catch (...) {
				throw 2;
			}

		} else {
			break;
		}
	}

	inputStream.close();

	if (currentRow < (size - 1)) {
		throw 1;
	}
	return list;
}
