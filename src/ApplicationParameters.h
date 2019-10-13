/*
 * ApplicationParameters.h
 *
 *  Created on: Feb 13, 2015
 *      Author: ubuntu
 */

#ifndef APPLICATIONPARAMETERS_H_
#define APPLICATIONPARAMETERS_H_
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

class ApplicationParameters {

private:
	string xPath;
	string yPath;
	string tPath;
	int rowCount;
	int colCount;
	int numberOfLabels;
	int totalLayerCount;
	int hiddenLayerSize;
	int numberOfThreads;
	int maxIteration;
	double lambda;
	int predict;
	int loadThetas;
	int saveThetas;
	int scale;
	int validCount;
	int valid;
	int testPercentage;


	void validateInputs(int argc, char **argv);
public:
	ApplicationParameters(int argc, char **argv);
	virtual ~ApplicationParameters();
	/*
	 * Get path of input list
	 */
	string getXPath();
	/*
	 * Get path of expectation list
	 */
	string getYPath();
	/*
	 * get path of previously saved thetas
	 */
	string getThetasPat();
	/*
	 * Get row count of X or Y list
	 */
	int getRowCount();
	/*
	 * Get column count of X list
	 */
	int getColumnCount();
	/*
	 * Get number of labels (Quantitiy of expectations)
	 */
	int getNumberOfLabels();
	/*
	 * Get total layer count
	 */
	int getTotalLayerCount();
	/*
	 * Get hidden layer size
	 */
	int getHiddenLayerSize();
	/*
	 * Get number of threads
	 */
	int getNumberOfThreads();
	/*
	 * Get maximum iterations
	 */
	int getMaxIteration();
	/*
	 * Get momentum
	 */
	int getLambda();
	/*
	 * Get if user ask application to\n
	 * do prediction after training complete
	 */
	int isCrossPredictionEnabled();
	int loadThetasEnabled();
	int saveThetasEnabled();
	int scaleInputsEnabled();
	void printHelp();
	int isValid();

	int getTestPercentage();

	void setRowCount(int count);

};

#endif /* APPLICATIONPARAMETERS_H_ */
