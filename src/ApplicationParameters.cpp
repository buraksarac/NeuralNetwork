/*
 * ApplicationParameters.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: ubuntu
 */

#include "ApplicationParameters.h"
#include "IOUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * Pojo class to hold application parameters and validate
 */
ApplicationParameters::ApplicationParameters(int argc, char **argv) {

	validateInputs(argc, argv);

}

ApplicationParameters::~ApplicationParameters() {
	// TODO Auto-generated destructor stub
}

void ApplicationParameters::printHelp() {
	printf("\nUSAGE:\n");
	printf("\n--help\tThis help info\n");
	printf("\n-x\tX(input) file path\n");
	printf("\n-y\tY(expected result) file path\n");
	printf("\n-r\tRowcount of X or Y file (should be equal)\n");
	printf("\n-c\tColumn count of X file (each row should have same count)\n");
	printf("\n-n\tNumber of labels in Y file (how many expected result)\n");
	printf("\n-t\tTotal layer count for neural network(including X)\n");
	printf("\n-h\tHidden layer size (excluding bias unit)\n");
	printf("\n-j\tNumber of cores(threads) on host pc\n");
	printf("\n-i\tNumber of iteration for training\n");
	printf("\n-l\tLambda value\n");
	printf("\n-f\tScale inputs for featured list\n");
	printf("\n-p\tDo prediction for each input after training complete (0 for disable 1 for enable default 1)\n");
	printf("\n-tp\tTheta path. If you have previously saved a prediction result you can continue"
			"\n\tfrom this result by loading from file path. (-lt value should be 1)\n");
	printf("\n-lt\tLoad previously saved thetas (prediction result)"
			"\n\t(0 for disable 1 for enable default 0) (-tp needs to be set)\n");
	printf("\n-st\tSave thetas (prediction result)(0 for disable 1 for enable default 1)\n");
	printf("\n");
	printf("\nPlease see http://www.u-db.org for more details\n");
}
void ApplicationParameters::validateInputs(int argc, char **argv) {

	//set default values
	this->numberOfThreads = 1;
	this->maxIteration = 1;
	this->lambda = 1;
	this->predict = 1;
	this->loadThetas = 0;
	this->saveThetas = 1;
	this->scale = 0;
	this->validCount = 0;
	this->valid = 1;
	//Check param size is a odd value
	if ((argc % 1) != 0) {
		printf("Invalid parameter size");
		this->valid = 0;
	}

	//validate and set inputs
	for (int i = 1; i < argc; i = i + 2) {
		if (!strcmp(argv[i], "--help")) {

			printHelp(); //print help

			this->valid = 0;
		} else if (!strcmp(argv[i], "-x")) {

			this->xPath = argv[i + 1]; //input path

			if (!IOUtils::fileExist(this->xPath)) { //check if file exist
				printf("-x parameter %s file doesnt exist!", this->xPath.c_str());
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-y")) {

			this->yPath = argv[i + 1]; //expectation list path

			if (!IOUtils::fileExist(this->yPath)) { // check if file exist

				printf("-y parameter %s file doesnt exist!", this->yPath.c_str());
				this->valid = 0;
			}
			this->validCount++;

		} else if (!strcmp(argv[i], "-r")) {

			this->rowCount = atoi(argv[i + 1]); //row count of x or y list

			if (this->rowCount < 10) { // minimum 10 row

				printf("Rowcount two small");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-c")) {
			this->colCount = atoi(argv[i + 1]);
			if (this->colCount < 1) {
				printf("Column count (-c) two small");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-n")) {
			this->numberOfLabels = atoi(argv[i + 1]);
			if (this->numberOfLabels < 2) {
				if (this->rowCount < 1) {
					printf("Number of labels two small");
					this->valid = 0;
				}
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-t")) {

			this->totalLayerCount = atoi(argv[i + 1]);

			if (this->totalLayerCount < 3) {

				printf("Total layer count should be greater than 2");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-h")) {
			this->hiddenLayerSize = atoi(argv[i + 1]);
			if (hiddenLayerSize < 3) {
				printf("hiddenLayerSize should be greater than 2");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-j")) {
			this->numberOfThreads = atoi(argv[i + 1]);
			if (this->numberOfThreads < 1) {
				printf("Wrong thread set");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-i")) {
			this->maxIteration = atoi(argv[i + 1]);
			if (this->maxIteration < 1) {
				printf("Wrong maxIteration set");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-f")) {
			this->scale = atoi(argv[i + 1]);
			if (!(this->scale == 0 || this->scale == 1)) {
				printf("Scale should be 1 or 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-l")) {
			this->lambda = atof(argv[i + 1]);
			if (!(this->lambda >= 0 && this->lambda <= 1)) {
				printf("Lambda should be between 1 and 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-p")) {
			this->predict = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-tp")) {
			this->tPath = argv[i + 1];
			if (!IOUtils::fileExist(this->tPath)) {
				printf("-t parameter %s file doesnt exist!", tPath.c_str());
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-lt")) {
			this->loadThetas = atoi(argv[i + 1]);
			if (!(this->loadThetas == 0 || this->loadThetas == 1)) {
				printf("loadThetas should be 1 or 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-st")) {
			this->saveThetas = atoi(argv[i + 1]);
			if (!(this->saveThetas == 0 || this->saveThetas == 1)) {
				printf("saveThetas should be 1 or 0");
				this->valid = 0;
			}
		} else {
			printf("Couldnt recognize user input");
			this->valid = 0;
		}

	}

	//make sure all 7 required params set
	if (this->validCount != 7) {
		printf("Bad parameters. You need to set all required parameters!");
		this->valid = 0;
	}

}

string ApplicationParameters::getXPath() {
	return this->xPath;
}

string ApplicationParameters::getYPath() {
	return this->yPath;
}

string ApplicationParameters::getThetasPat() {
	return this->tPath;
}

int ApplicationParameters::getRowCount() {
	return this->rowCount;
}

int ApplicationParameters::getColumnCount() {
	return this->colCount;
}

int ApplicationParameters::getNumberOfLabels() {
	return this->numberOfLabels;
}

int ApplicationParameters::getNumberOfThreads() {
	return this->numberOfThreads;
}

int ApplicationParameters::getTotalLayerCount() {
	return this->totalLayerCount;
}

int ApplicationParameters::getHiddenLayerSize() {
	return this->hiddenLayerSize;
}

int ApplicationParameters::getMaxIteration() {
	return this->maxIteration;
}

int ApplicationParameters::getLambda() {
	return this->lambda;
}

int ApplicationParameters::isCrossPredictionEnabled() {
	return this->predict;
}

int ApplicationParameters::loadThetasEnabled() {
	return this->loadThetas;
}

int ApplicationParameters::saveThetasEnabled() {
	return this->saveThetas;
}

int ApplicationParameters::scaleInputsEnabled() {
	return this->scale;
}

int ApplicationParameters::isValid() {
	return this->valid;
}

