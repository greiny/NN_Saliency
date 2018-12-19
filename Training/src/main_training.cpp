/*******************************************************************
* Neural Network Training Example
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/
//standard libraries
#include <iostream>
#include <ctime>
#include <sstream>
#include <fstream>

//custom includes
#include "modules/matrix_maths.h"
#include "modules/neuralNetwork.h"
#include "modules/neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

int main()
{		
	//seed random number generator
	srand( (unsigned int) time(0) );
	
	//Training condition
	int max_epoch = 10000000;
	double accuracy = 99.9;
	double max_time = 3000;
	float lr = 0.001;
	float momentum = 0.9;
	float tRatio = 0.8;
	float vRatio = 0.2;

	//Layer Construction
	vector<ConvLayer> CLayer;
	vector<int> FCLayer{128};

	// for data
	int nInput = 25*25*3;
	int nOutput = 3;

	//create data set reader and load data file
	dataReader d;
	d.loadDataFile4Train("imgdata.txt",nInput,nOutput,tRatio,vRatio,NML);
	d.setCreationApproach( STATIC, 1 );

	//create neural network
	neuralNetwork nn(nInput,FCLayer,nOutput);

	//save the Training Condition
	char file_name[255];
	int file_no = 0;
	sprintf(file_name,"log/condition%d.csv",file_no);
	for ( int i=0 ; i < 100 ; i++ )
	{
		ifstream test(file_name);
		if (!test) break;
		file_no++;
		sprintf(file_name,"log/condition%d.csv",file_no);
	}

	//create neural network trainer and save log
	neuralNetworkTrainer nT( &nn );
	nT.setTrainingParameters(lr, momentum, true);
	nT.setStoppingConditions(max_epoch, accuracy, max_time);

	ofstream logTrain;
	logTrain.open(file_name,ios::out);
	if ( logTrain.is_open() )
	{
		logTrain << "Input" << "," << nInput << endl;
		if(CLayer.size()!=0){ 
			logTrain << "ConvLayer" << "," << CLayer.size() << endl;
			for (int i=0; i<CLayer.size(); i++)
				logTrain << "Kernel["<< i << "]," << CLayer[i].nKernel << "," << CLayer[i].sKernel << "x" << CLayer[i].sKernel << ","<< CLayer[i].pdim << endl;
		}
		logTrain << "FCLayer" << "," << FCLayer.size() << endl;
		for (int i=0; i<FCLayer.size(); i++) logTrain << "Neuron["<< i << "]," << FCLayer[i] << endl;
		logTrain << "Output" << "," << nOutput << endl;
		logTrain << "TrainingSet" << "," << (int)d.trainingDataEndIndex << endl;
		logTrain << "ValidationSet" << "," << (int)((d.trainingDataEndIndex/tRatio)*vRatio) << endl;
		logTrain << "Accuracy(%)" << "," << accuracy << endl;
		logTrain << "Learning_rate" << "," << lr << endl;
		logTrain << "Momentum" << "," << momentum << endl;
		logTrain.close();
	}

	ostringstream log_name;
	log_name << "log/log" << file_no << ".csv";
	const char* log_name_char = new char[log_name.str().length()+1];
	log_name_char = log_name.str().c_str();
	nT.enableLogging(log_name_char, 10);

	char w_name[255];
	sprintf(w_name,"log/weights%d.txt",file_no);
	nn.enableLoggingWeight(w_name); // ((#input)*(#neuron)+(#neuron)*(#output)--Weight)+((#neuron+#output)--Bias)

	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}

	CLayer.clear();
	//print success
	cout << endl << "Neuron weights saved to '" << w_name << "'" << endl;
	cout << endl << endl << "-- END OF PROGRAM --" << endl;
} 
