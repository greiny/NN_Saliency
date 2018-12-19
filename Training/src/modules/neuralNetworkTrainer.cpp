//standard includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>

//include definition file
#include "neuralNetworkTrainer.h"

using namespace std;

/*******************************************************************
* constructor
********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer( neuralNetwork *nn )	:	NN(nn),
																	epoch(0),
																	learningRate(LEARNING_RATE),
																	momentum(MOMENTUM),
																	maxEpochs(MAX_EPOCHS),
																	desiredAccuracy(DESIRED_ACCURACY),																	
																	useBatch(false),
																	trainingSetAccuracy(0),
																	validationSetAccuracy(0),
																	generalizationSetAccuracy(0),
																	trainingSetMSE(0),
																	validationSetMSE(0),
																	generalizationSetMSE(0)																	
{
	//create delta lists
	//--------------------------------------------------------------------------------------------------------
	if (NN->nLayer==0) {
		deltaInputOutput = new( double*[NN->nInput] );
		for ( int i=0; i < NN->nInput; i++ ) 
		{
			deltaInputOutput[i] = new (double[NN->nOutput]);
			for ( int j=0; j < NN->nOutput; j++ ) deltaInputOutput[i][j] = 0;
		}
	}
	else {
		deltaInputHidden = new( double*[NN->nInput + 1] );
		for ( int i=0; i <= NN->nInput; i++ ) 
		{
			deltaInputHidden[i] = new (double[NN->nHidden[0]]);
			for ( int j=0; j < NN->nHidden[0]; j++ ) deltaInputHidden[i][j] = 0;
		}

		if (NN->nLayer>1)
		{
			for (int k=0; k<(NN->nLayer-1); k++)
			{
				deltaHiddenHidden.push_back(new( double*[NN->nHidden[k]+1])); // hidden(k)-hidden(k+1)
				for ( int i=0; i <= NN->nHidden[k]; i++ )
				{
					deltaHiddenHidden[k][i] = new (double[NN->nHidden[k+1]]);
					for ( int j=0; j < NN->nHidden[k+1]; j++ ) deltaHiddenHidden[k][i][j] = 0;
				}
			}
		}

		deltaHiddenOutput = new( double*[NN->nHidden[NN->nLayer-1] + 1] ); // +1 means bias neuron
		for ( int i=0; i <= NN->nHidden[NN->nLayer-1]; i++ )
		{
			deltaHiddenOutput[i] = new (double[NN->nOutput]);
			for ( int j=0; j < NN->nOutput; j++ ) deltaHiddenOutput[i][j] = 0;
		}
	}

	//create error gradient storage
	for (int k=0; k< NN->nLayer; k++)
	{
		hiddenErrorGradients.push_back(new(double[NN->nHidden[k]])); // number of HiddenNeuron and bias
		for ( int i=0; i < NN->nHidden[k]; i++ ) hiddenErrorGradients[k][i] = 0;
	}
	outputErrorGradients = new( double[NN->nOutput] );
	for ( int i=0; i < NN->nOutput; i++ ) outputErrorGradients[i] = 0;
}


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( double lR, double m, bool batch )
{
	learningRate = lR;
	momentum = m;
	useBatch = batch;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, double dAccuracy, double endtime )
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;
	maxTime = endtime;	
}
/*******************************************************************
* Enable training logging
********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
	//create log file 
	if ( ! logFile.is_open() )
	{
		logFile.open(filename, ios::out);

		if ( logFile.is_open() )
		{
			//write log file header
			logFile << "Epoch, Training Set Accuracy, Generalization Set Accuracy, Validation Set Accuracy, Training Set MSE, Generalization Set MSE, Validation Set MSE, " << endl;
			
			//enable logging
			loggingEnabled = true;
			
			//resolution setting;
			logResolution = resolution;
			lastEpochLogged = -resolution;
		}
	}
}
/*******************************************************************
* calculate output error gradient
********************************************************************/
inline double neuralNetworkTrainer::getOutputErrorGradient( double desiredValue, double outputValue)
{
	//return error gradient
	return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

/*******************************************************************
* calculate input error gradient
********************************************************************/
double neuralNetworkTrainer::getHiddenErrorGradient( int k, int j ) // k:#layer j:#neuron[k]
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	if (k==(NN->nLayer-1))
		for( int i = 0; i < NN->nOutput; i++ ) weightedSum += NN->wHiddenOutput[j][i] * outputErrorGradients[i];
	else
		for( int i = 0; i < NN->nHidden[k+1]; i++ ) weightedSum += NN->wHiddenHidden[k][j][i] * hiddenErrorGradients[k+1][i];
	//return error gradient
	return NN->hiddenNeurons[k][j] * ( 1 - NN->hiddenNeurons[k][j] ) * weightedSum;
}
/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void neuralNetworkTrainer::trainNetwork( trainingDataSet* tSet )
{
	cout	<< endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
			<< " " << NN->nInput << " Input Neurons, " << NN->nLayer << " Hidden Layers, " << NN->nOutput << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;
	lastEpochLogged = -logResolution;

	//Time checking start!
	float time = 0;
	auto t0 = std::chrono::high_resolution_clock::now();
	double hTAcc = 0;
	double hGAcc = 0;
	long hEpoch = 0;
	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while (	( generalizationSetAccuracy <  desiredAccuracy ) && epoch < maxEpochs && time < maxTime )
	{			
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch( tSet->trainingSet );

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet );
		generalizationSetMSE = NN->getSetMSE( tSet->generalizationSet );
		//NN->getRegression( tSet->generalizationSet,1 );

		//get validation set accuracy and MSE
		validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
		validationSetMSE = NN->getSetMSE(tSet->validationSet);
		//NN->getRegression( tSet->validationSet,2 );

		//Log Training results
		if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
		{
			logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << validationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << ","  << validationSetMSE << endl;
			lastEpochLogged = epoch;
		}
		
		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		auto t1 = std::chrono::high_resolution_clock::now();
		if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
		{	
			t0 = t1;
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
		}
		if ( hTAcc < trainingSetAccuracy  )
		{
			hTAcc = trainingSetAccuracy;
			hGAcc = generalizationSetAccuracy;
			hEpoch = epoch;
			NN->saveWeights(hTAcc,hGAcc,hEpoch);
		}

		time = std::chrono::duration<float>(t1-t0).count();

		//once training set is complete increment epoch
		epoch++;

	}//end while

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << validationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << "," << validationSetMSE << endl;

	//out validation accuracy and MSE
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << hEpoch << endl;
	cout << " Training Set Accuracy: " << hTAcc << endl;
	cout << " Generalization Set Accuracy: " << hGAcc << endl;
}

/*******************************************************************
* Run a single training epoch
********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet )
{
	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;

	//for every training pattern
	for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
	{
		//feed inputs through network and backpropagation errors
		NN->feedForward( trainingSet[tp]->pattern );
		backpropagate( trainingSet[tp]->target );	

		//pattern correct flag
		bool patternCorrect = true;
#if 1
		double* new_outputNeurons = NN->sotfMax( NN->outputNeurons);
		//check all outputs from neural network against desired values
		for ( int k = 0; k < NN->nOutput; k++ )
		{					
			//pattern incorrect if desired and output differ
			if ( new_outputNeurons[k] != trainingSet[tp]->target[k] ) patternCorrect = false;
			//calculate MSE
			mse += pow(( NN->outputNeurons[k] - trainingSet[tp]->target[k] ), 2);
		}
		free(new_outputNeurons);
#else
		//check all outputs from neural network against desired values
		for ( int k = 0; k < NN->nOutput; k++ )
		{					
			//pattern incorrect if desired and output differ
			if ( NN->clampOutput( NN->outputNeurons[k] ) != trainingSet[tp]->target[k] ) patternCorrect = false;
			//calculate MSE
			mse += pow(( NN->outputNeurons[k] - trainingSet[tp]->target[k] ), 2);
		}	
#endif

		//if pattern is incorrect add to incorrect count
		if ( !patternCorrect ) incorrectPatterns++;
	
	}//end for

	//if using batch learning - update the weights
	if ( useBatch ) updateWeights();
	
	//update training accuracy and MSE
	trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
	trainingSetMSE = mse / ( NN->nOutput * trainingSet.size() );
	//NN->getRegression(trainingSet,0);
}
/*******************************************************************
* Propagate errors back through NN and calculate delta values
********************************************************************/
void neuralNetworkTrainer::backpropagate( double* desiredOutputs )
{	
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
	if (NN->nLayer==0) {
		for (int j = 0; j < NN->nOutput; j++)
		{
			//get error gradient for every output node
			outputErrorGradients[j] = getOutputErrorGradient( desiredOutputs[j], NN->outputNeurons[j] );
		
			//for all nodes in hidden layer and bias neuron
			for (int i = 0; i < NN->nInput; i++)
			{				
				//calculate change in weight
				if ( !useBatch ) deltaInputOutput[i][j] = learningRate * NN->inputNeurons[i] * outputErrorGradients[j] + momentum * deltaInputOutput[i][j];
				else deltaInputOutput[i][j] += learningRate * NN->inputNeurons[i] * outputErrorGradients[j];
			}
		}
	}
	else {
		for (int j = 0; j < NN->nOutput; j++)
		{
			//get error gradient for every output node
			outputErrorGradients[j] = getOutputErrorGradient( desiredOutputs[j], NN->outputNeurons[j] );
		
			//for all nodes in hidden layer and bias neuron
			for (int i = 0; i <= NN->nHidden[NN->nLayer-1]; i++)
			{				
				//calculate change in weight
				if ( !useBatch ) deltaHiddenOutput[i][j] = learningRate * NN->hiddenNeurons[NN->nLayer-1][i] * outputErrorGradients[j] + momentum * deltaHiddenOutput[i][j];
				else deltaHiddenOutput[i][j] += learningRate * NN->hiddenNeurons[NN->nLayer-1][i] * outputErrorGradients[j];
			}
		}

		//modify deltas between hidden and hidden layers
		//--------------------------------------------------------------------------------------------------------
		if (NN->nLayer>1)
		{
			for (int k=NN->nLayer-2; k>=0; k--)
			{
				for (int j = 0; j < NN->nHidden[k+1]; j++)
				{
					//get error gradient for every hidden node
					hiddenErrorGradients[k+1][j] = getHiddenErrorGradient(k+1,j);

					//for all nodes in input layer and bias neuron
					for (int i = 0; i <= NN->nHidden[k]; i++)
					{
						//calculate change in weight
						if ( !useBatch ) deltaHiddenHidden[k][i][j] = learningRate * NN->hiddenNeurons[k][i] * hiddenErrorGradients[k+1][j] + momentum * deltaHiddenHidden[k][i][j];
						else deltaHiddenHidden[k][i][j] += learningRate * NN->hiddenNeurons[k][i] * hiddenErrorGradients[k+1][j];
					}
				}
			}
		}

		//modify deltas between input and hidden layers
		//--------------------------------------------------------------------------------------------------------
		for (int j = 0; j < NN->nHidden[0]; j++)
		{
			//get error gradient for every hidden node
			hiddenErrorGradients[0][j] = getHiddenErrorGradient(0,j);
			//for all nodes in input layer and bias neuron
			for (int i = 0; i <= NN->nInput; i++)
			{
				//calculate change in weight 
				if ( !useBatch ) deltaInputHidden[i][j] = learningRate * NN->inputNeurons[i] * hiddenErrorGradients[0][j] + momentum * deltaInputHidden[i][j];
				else deltaInputHidden[i][j] += learningRate * NN->inputNeurons[i] * hiddenErrorGradients[0][j];
			}
		}
	}
	
	//if using stochastic learning update the weights immediately
	if ( !useBatch ) updateWeights();
}
/*******************************************************************
* Update weights using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeights()
{
	if (NN->nLayer==0) {
		//input -> output weights
		for (int i = 0; i < NN->nInput; i++)
		{
			for (int j = 0; j < NN->nOutput; j++)
			{
				//update weight
				NN->wInputOutput[i][j] += deltaInputOutput[i][j];	
				//clear delta only if using batch (previous delta is needed for momentum
				if (useBatch) deltaInputOutput[i][j] = 0;				
			}
		}		
	}
	else {
		//input -> hidden weights
		for (int i = 0; i <= NN->nInput; i++)
		{
			for (int j = 0; j < NN->nHidden[0]; j++)
			{
				//update weight
				NN->wInputHidden[i][j] += deltaInputHidden[i][j];	
				//clear delta only if using batch (previous delta is needed for momentum
				if (useBatch) deltaInputHidden[i][j] = 0;				
			}
		}

		//hidden[k] -> hidden[k+1] weights
		if (NN->nLayer>1)
		{
			for (int k=0; k<NN->nLayer-1; k++)
			{
				for ( int i=0; i <= NN->nHidden[k]; i++ )
				{
					for ( int j=0; j < NN->nHidden[k+1]; j++ )
					{
						NN->wHiddenHidden[k][i][j] += deltaHiddenHidden[k][i][j];
						if (useBatch)deltaHiddenHidden[k][i][j] = 0;
					}
				}
			}
		}

		//hidden -> output weights
		for (int i = 0; i <= NN->nHidden[NN->nLayer-1]; i++)
		{
			for (int j = 0; j < NN->nOutput; j++)
			{
				//update weight
				NN->wHiddenOutput[i][j] += deltaHiddenOutput[i][j];
				//clear delta only if using batch (previous delta is needed for momentum)
				if (useBatch)deltaHiddenOutput[i][j] = 0;
			}
		}
	}
}
