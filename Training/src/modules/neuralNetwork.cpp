//standard includes

#include "neuralNetwork.h"

using namespace std;
using namespace cv;

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nInput, vector<int> nHidden, int nOutput) : nInput(nInput), nHidden(nHidden), nOutput(nOutput), nLayer(nHidden.size())
{
	int nLayer = nHidden.size();
	
	//create neuron lists
	inputNeurons = new( double[nInput + 1] );
	for ( int i=0; i <= nInput; i++ ) inputNeurons[i] = 0;
	inputNeurons[nInput] = 1;

	outputNeurons = new( double[nOutput] );
	for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

	for (int k=0; k<nLayer; k++)
	{
		hiddenNeurons.push_back(new(double[nHidden[k]+1])); // number of HiddenNeuron and bias
		for ( int i=0; i < nHidden[k]; i++ ) hiddenNeurons[k][i] = 0;
		hiddenNeurons[k][nHidden[k]] = -1; // order of bias
	}

	//create weight lists (include bias neuron weights)
	if (nLayer==0) {
		wInputOutput = new( double*[nInput] );
		for ( int i=0; i < nInput; i++ )
		{
			wInputOutput[i] = new (double[nOutput]);
			for ( int j=0; j < nOutput; j++ ) wInputOutput[i][j] = 0;
		}
	}
	else {
		wInputHidden = new( double*[nInput + 1] );
		for ( int i=0; i <= nInput; i++ )
		{
			wInputHidden[i] = new (double[nHidden[0]]);
			for ( int j=0; j < nHidden[0]; j++ ) wInputHidden[i][j] = 0;
		}

		if (nLayer>1)
		{
			for (int k=0; k<nLayer-1; k++)
			{
				wHiddenHidden.push_back(new( double*[nHidden[k] + 1])); // hidden(k)-hidden(k+1)
				for ( int i=0; i <= nHidden[k]; i++ )
				{
					wHiddenHidden[k][i] = new (double[nHidden[k+1]]);
					for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = 0;
				}
			}
		}
		wHiddenOutput = new( double*[nHidden[nLayer-1] + 1] );
		for ( int i=0; i <= nHidden[nLayer-1]; i++ )
		{
			wHiddenOutput[i] = new (double[nOutput]);
			for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;
		}
	}

	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] outputNeurons;
	for (int k=0; k<nLayer; k++) delete[] hiddenNeurons[k];

	//delete weight storage
	if (nLayer==0){
		for (int i=0; i < nInput; i++) delete[] wInputOutput[i];
		delete[] wInputOutput;
	}
	else{
		for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
		delete[] wInputHidden;
		for (int j=0; j <= nHidden[nLayer-1]; j++) delete[] wHiddenOutput[j];
		delete[] wHiddenOutput;
		if (nLayer>1) for (int k=0; k<nLayer-1; k++) for (int i=0; i<=nHidden[k]; i++) delete[] wHiddenHidden[k][i];
	}
}

/*******************************************************************
* Load Neuron Weights
********************************************************************/
bool neuralNetwork::loadWeights(const char* filename)
{
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		vector<double> weights;
		string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) 
			{				
				//store inputs		
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				int i = 0;
				t=strtok (cstr,",");
				
				while ( t!=NULL )
				{	
					weights.push_back( atof(t) );
				
					//move token onwards
					t = strtok(NULL,",");
					i++;			
				}

				//free memory
				delete[] cstr;
			}
		}

		if (nLayer==0) {
			int num_weight = (nInput) * nOutput;
			if ( weights.size() != num_weight )
			{
				cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;
				cout << "System requires " << num_weight << " and " << weights.size() << " put in " << endl;
				inputFile.close();
				return false;
			}
			else
			{
				//set weights
				int pos = 0;
				for ( int i=0; i <= nInput; i++ )
					for ( int j=0; j < nOutput; j++ ) wInputOutput[i][j] = weights[pos++];
			}
			//print success
			cout << endl << "Neuron weights loaded successfully from '" << filename << "'" << endl;
			inputFile.close();
			return true;
		}
		else { 
			//check if sufficient weights were loaded
			int num_weight = (nInput + 1) * nHidden[0] + (nHidden[nLayer-1] +  1) * nOutput;
			if (nLayer>1) for (int k=0; k<nLayer-1; k++) num_weight += (nHidden[k] +  1) * nHidden[k+1];

			if ( weights.size() != num_weight )
			{
				cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;
				cout << "System requires " << num_weight << " and " << weights.size() << " put in " << endl;
				//close file
				inputFile.close();

				return false;
			}
			else
			{
				//set weights
				int pos = 0;

				for ( int i=0; i <= nInput; i++ )
				{
					for ( int j=0; j < nHidden[0]; j++ ) wInputHidden[i][j] = weights[pos++];
				}
				if (nLayer>1)
				{
					for (int k=0; k<nLayer-1; k++)
					{
						for ( int i=0; i <= nHidden[k]; i++ )
						{
							for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = weights[pos++];
						}
					}
				}
				for ( int i=0; i <= nHidden[nLayer-1]; i++ )
				{
					for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = weights[pos++];
				}
		}
			//print success
			cout << endl << "Neuron weights loaded successfully from '" << filename << "'" << endl;
			//close file
			inputFile.close();
			
			return true;
		}	
	
	}
	else 
	{
		cout << endl << "Error - Weight input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}

/*******************************************************************
* Save Neuron Weights
********************************************************************/
void neuralNetwork::enableLoggingWeight(const char* filename)
{
	//open file for reading
	outputFilename = filename;
	loggingEnabled = true;
}

bool neuralNetwork::saveWeights(double hTAcc, double hGAcc, long hEpoch)
{
	outputFile.open(outputFilename, ios::out);
	if ( outputFile.is_open() && loggingEnabled )
	{
		outputFile.precision(10); // # of floating point
		if (nLayer==0) {
			for ( int i=0; i <nInput; i++ )  // '=' means loop including bias neuron
				for ( int j=0; j < nOutput; j++ ) outputFile << wInputOutput[i][j] << ","; // (nInput+1)*nHidden
			outputFile << endl;
		}
		else {
			//output weights
			for ( int i=0; i <= nInput; i++ )  // '=' means loop including bias neuron
				for ( int j=0; j < nHidden[0]; j++ ) outputFile << wInputHidden[i][j] << ","; // (nInput+1)*nHidden
			outputFile << endl;
			if (nLayer>1)
			{
				for (int k=0; k<nLayer-1; k++)
				{
					for ( int i=0; i <= nHidden[k]; i++ )
					{
						for ( int j=0; j < nHidden[k+1]; j++ ) outputFile << wHiddenHidden[k][i][j] << ",";
					}
					outputFile << endl;
				}
			}
			for ( int i=0; i <= nHidden[nLayer-1]; i++ ) // '=' means loop including bias neuron
			{
				for ( int j=0; j < nOutput; j++ )
				{
					outputFile << wHiddenOutput[i][j];	// (nHidden+1)*nOutput
					if ( i * nOutput + j + 1 != (nHidden[nLayer-1] + 1) * nOutput ) outputFile << ",";
				}
			}
		}
		outputFile << endl << "hTAcc," << hTAcc << ",hGAcc," << hGAcc << ",hEpoch," << hEpoch << ",";
		//close file
		outputFile.close();

		return true;
	}
	else
	{
		cout << endl << "Error - Weight output file '" << outputFilename << "' could not be created: " << endl;
		return false;
	}
}
/*******************************************************************
* Feed pattern through network and return results
********************************************************************/
double* neuralNetwork::feedForwardPattern(double *pattern)
{
	feedForward(pattern);
	//create copy of output results
	double* results = new double[nOutput];
	//for (int i=0; i < nOutput; i++ ) results[i] = clampOutput(outputNeurons[i]);
	for (int i=0; i < nOutput; i++ ) results[i] = outputNeurons[i];
	return results;
}
/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//correct pattern flag
		bool correctResult = true;
#if 1
		double* new_outputNeurons = sotfMax(outputNeurons);
		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
			if ( new_outputNeurons[k] != set[tp]->target[k] ) correctResult = false;
		free(new_outputNeurons);
#else
		for ( int k = 0; k < nOutput; k++ )		
			//set flag to false if desired and output differ
			if ( clampOutput(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;

#endif
		//inc training error for a incorrect result
		if ( !correctResult ) incorrectResults++;	
		
	}//end for
	
	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}

/*******************************************************************
* Return the NN target and output
********************************************************************/

void neuralNetwork::getRegression( std::vector<dataEntry*>& set , int i)
{
	ofstream logReg;
	if (i==0) logReg.open("log/logReg_TrainingtSet.csv",ios::out);
	if (i==1) logReg.open("log/logReg_GeneralSet.csv",ios::out);
	if (i==2) logReg.open("log/logReg_TestSet.csv",ios::out);
	if ( logReg.is_open() )
	{
		logReg << "Output" << "," << "Target" << endl;

		//for every training input array
		for ( int tp = 0; tp < (int) set.size(); tp++)
		{
			//feed inputs through network and backpropagate errors
			feedForward( set[tp]->pattern );
			//check all outputs against desired output values
			for ( int k = 0; k < nOutput; k++ )
			{
				logReg << outputNeurons[k] << "," << set[tp]->target[k]<< endl;
			}
		}//end for
	}
	logReg.close();
}

/*******************************************************************
* Return the NN mean squared error on the set
********************************************************************/
double neuralNetwork::getSetMSE( std::vector<dataEntry*>& set )
{
	double mse = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
		{					
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}		
		
	}//end for
	
	//calculate error and return as percentage
	return mse/(nOutput * set.size());
}


/*******************************************************************
* Initialize Neuron Weights
********************************************************************/
void neuralNetwork::initializeWeights()
{
	//set range
	double rH = 1/sqrt((double)nInput);
	vector<double> rO;
	for (int k=0; k<nLayer; k++) rO.push_back(1/sqrt((double)nHidden[k]));

	//set weights
	if (nLayer==0) {
		for(int i=0; i<nInput; i++)
			for(int j=0; j<nOutput; j++) wInputOutput[i][j] = (((double)(rand()%100)+1)/100*2*rH)-rH;

	}
	else {
		for(int i=0; i<=nInput; i++)
			for(int j=0; j<nHidden[0]; j++) wInputHidden[i][j] = (((double)(rand()%100)+1)/100*2*rH)-rH;

		if (nLayer>1)
		{
			for (int k=0; k<nLayer-1; k++)
			{
				for ( int i=0; i <= nHidden[k]; i++ )
				{
					for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = (((double)(rand()%100)+1)/100 * 2 * rO[k] ) - rO[k];
				}
			}
		}

		for(int i = 0; i <= nHidden[nLayer-1]; i++)
			for(int j = 0; j < nOutput; j++) wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO[nLayer-1] ) - rO[nLayer-1];
	}
}
/*******************************************************************
* Activation Function
********************************************************************/
inline double neuralNetwork::activationFunction( double x, int method)
{
	if (method==NL_SIGMOID) return 1/(1+exp(-x));
	else if (method==NL_TANH) return tanh(x);
	else if (method==NL_RELU) return max((double)0,x);
	else return -1;
}	
/*******************************************************************
* Output Clamping
********************************************************************/
int neuralNetwork::clampOutput( double x )
{
	if ( x > 0.9 ) return 1;
	else if ( x < 0.1 ) return 0;
	else return -1;
}

double* neuralNetwork::sotfMax( double* x )
{
	double max_val=0;
	int max_loc=0;
	for(int k=0; k < nOutput; k++) {
		if(max_val<x[k]) {
			max_val = x[k]; max_loc = k;
		}
	}
	double* new_output = new double[nOutput];
	for(int k=0; k < nOutput; k++){
		if (k==max_loc) new_output[k] = 1;
		else new_output[k] = 0;
	}
	return new_output;
}
/*******************************************************************
* Feed Forward Operation
********************************************************************/

void neuralNetwork::feedForward(double* pattern)
{
	//set input neurons to input values
	for(int i = 0; i < nInput; i++) inputNeurons[i] = pattern[i];
	if (nLayer==0) {
		for(int j=0; j < nOutput; j++)
		{
			//clear value
			outputNeurons[j] = 0;
			//get weighted sum of pattern and bias neuron
			for( int i=0; i < nInput; i++ ) outputNeurons[j] += inputNeurons[i] * wInputOutput[i][j];
			//set to result of sigmoid
			outputNeurons[j] = activationFunction( outputNeurons[j] , NL_SIGMOID);
		}
	}
	else {
		for(int j=0; j < nHidden[0]; j++)
		{
			//clear value
			hiddenNeurons[0][j] = 0;
			//get weighted sum of pattern and bias neuron
			for( int i=0; i <= nInput; i++ ) hiddenNeurons[0][j] += inputNeurons[i] * wInputHidden[i][j];
			//set to result of sigmoid
			hiddenNeurons[0][j] = activationFunction( hiddenNeurons[0][j], NL_SIGMOID );
		}

		if (nLayer>1)
		{
			for (int k=1; k<nLayer; k++)
			{
				//Calculate Hidden Layer values - include bias neuron
				for(int j=0; j < nHidden[k]; j++)
				{
					//clear value
					hiddenNeurons[k][j] = 0;
					//get weighted sum of pattern and bias neuron
					for( int i=0; i <= nHidden[k-1]; i++ )
					{
						hiddenNeurons[k][j] += hiddenNeurons[k-1][i] * wHiddenHidden[k-1][i][j];
					}
					//set to result of sigmoid
					hiddenNeurons[k][j] = activationFunction( hiddenNeurons[k][j], NL_SIGMOID );
				}
			}
		}

		for(int j=0; j < nOutput; j++)
		{
			//clear value
			outputNeurons[j] = 0;
			//get weighted sum of pattern and bias neuron
			for( int i=0; i <= nHidden[nLayer-1]; i++ ) outputNeurons[j] += hiddenNeurons[nLayer-1][i] * wHiddenOutput[i][j];
			//set to result of sigmoid
			outputNeurons[j] = activationFunction( outputNeurons[j], NL_SIGMOID );
		}
	}
}
