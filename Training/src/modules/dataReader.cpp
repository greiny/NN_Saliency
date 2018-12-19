//include definition file
#include "dataReader.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

/*******************************************************************
* Destructor
********************************************************************/
dataReader::~dataReader()
{
	//clear data
	CLayer.clear();
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();		 
}
/*******************************************************************
* Loads a csv file of input data
********************************************************************/
bool dataReader::loadDataFile4Train( const char* filename, int nI, int nT ,float tratio, float gratio, bool method)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();
	
	//set number of inputs and outputs
	nInputs = nI;
	nTargets = nT;

	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);	

	if ( inputFile.is_open() )
	{
		string line = "";

		//read data
		while ( !inputFile.eof() )
		{
			// get num of line
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) processLine(line);
		}	

		//print success
		cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << endl;

		if (method == NML){
			//normalize data
			double *buf[(int)nInputs];
			vector <double> minVal, maxVal;
			ofstream maxmin("log/maxmin.csv");
			for(int i=0; i < nInputs; i++)
			{
				buf[i] = new( double[(int)data.size()] );
				double min=100000, max = 0;
				for(int j=0; j < (int)data.size(); j++)
				{
				    double temp = (double)data[j]->pattern[i];
				    if (temp > max) max = temp;
				    else if (temp < min) min = temp;
				    buf[i][j] = temp;
				}
				minVal.push_back(min);
				maxVal.push_back(max);
				maxmin << min << "," << max << endl;
			}
			maxmin.close();
			for(int ii=0; ii < nInputs; ii++)
			{
				for(int jj=0; jj < data.size(); jj++)
				{
					double diff = maxVal[ii]-minVal[ii];
					if (diff==0) data[jj]->pattern[ii]=1.0;
					else data[jj]->pattern[ii]=(double)((buf[ii][jj]-minVal[ii])/diff); // normalization to 0~1
				}
			};
		}

		else{
			//standardize data
			double* avg = new double[nInputs];
			double* std = new double[nInputs];
			for(int i=0; i < nInputs; i++)
			{
				Mat buf = Mat::zeros(1,(int)data.size(),CV_64FC1);
				for(int j=0; j < (int)data.size(); j++) buf.ATD(0,j) = data[j]->pattern[i];
				Scalar mean, stddev;
				meanStdDev(buf,mean,stddev);
				avg[i]=mean[0]; 
				std[i]=stddev[0];
			}
			for(int ii=0; ii < nInputs; ii++)
			{
				for(int jj=0; jj < data.size(); jj++)
				{
					if (std[ii]!=0) data[jj]->pattern[ii]=(double)((data[jj]->pattern[ii]-avg[ii])/std[ii]);
					else data[jj]->pattern[ii]=(double)0.0f;
				}
			}
			delete[] avg;
			delete[] std;
		}

		//shuffle data
		random_shuffle(data.begin(), data.end());

		//split data set
		trainingDataEndIndex = (int) ( tratio * data.size() );
		int gSize = (int) ( ceil(gratio * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );

		//generalization set
		for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) tSet.generalizationSet.push_back( data[i] );

		//validation set
		for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

		//close file
		inputFile.close();

		return true;
	}
	else
	{
		cout << "Error Opening Input File: " << filename << endl;
		return false;
	}
}

bool dataReader::loadImageFile4Train( const char* filename, int nI, int nT ,float tratio, float gratio, bool GAP)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];
	data.clear();
	tSet.clear();

	//set number of inputs and outputs
	sImage = nI;
	nTargets = nT;

	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		string line = "";

		//read data
		while ( !inputFile.eof() )
		{
			// get num of line
			getline(inputFile, line);

			//process line
			if (line.length() > 2 ) processLine4Image(line);
		}

		//print success
		cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << endl;

		// normalize and convolve data
		for(int jj=0; jj < data.size(); jj++) data[jj]-> pattern = ConvNPooling(data[jj]->pattern,sImage,GAP);

		//shuffle data
		random_shuffle(data.begin(), data.end());

		//split data set
		trainingDataEndIndex = (int) ( tratio * data.size() );
		int gSize = (int) ( ceil(gratio * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );

		//generalization set
		for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) tSet.generalizationSet.push_back( data[i] );

		//validation set
		for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

		//close file
		inputFile.close();

		return true;
	}
	else
	{
		cout << "Error Opening Input File: " << filename << endl;
		return false;
	}
}

void dataReader::loadImage4Test(const Mat frame,int sI, int nT, bool GAP)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();

	sImage = sI;
	nTargets = nT;
	processMat(frame,GAP);

	//validation set
	trainingDataEndIndex = (int)0;
	for ( int i = 0; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

}

bool dataReader::maxmin( const char* filename)
{
		//open file for maxmin value
		fstream maxminFile;
		maxminFile.open(filename, ios::in);

		if ( maxminFile.is_open() )
		{
			string line2 = "";
			cout << "ManMin Reference File Loaded" << endl;

			//read data
			while ( !maxminFile.eof() )
			{
				// get num of line
				getline(maxminFile, line2);
				if (line2.length() > 2 ) processLine4maxmin(line2);  //ref[i]->p_min
			}

			//close file
			maxminFile.close();
			return true;
		}
		else
		{
			cout << "Error Opening MaxMin File " << endl;
			return false;
		}
}

/*******************************************************************
* Processes a single line from the data file
********************************************************************/

void dataReader::processMat( const Mat frame, bool GAP )
{
	//create new pattern and target
	double* pattern = new double[sImage];
	double* target = new double[nTargets];
	frame.convertTo(frame, CV_64FC1, 1.0/255, 0);
	pattern = ConvNPooling(frame,GAP);
	for (int i=0; i<nTargets; i++) if ( i < sImage ) target[i] = 0;

	//add to records
	data.push_back( new dataEntry( pattern, target ) );		
}


void dataReader::processLine( string &line )
{
	//create new pattern and target
	double* pattern = new double[nInputs];
	double* target = new double[nTargets];
	
	//store inputs		
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");
	
	while ( t!=NULL && i < (nInputs + nTargets) )
	{	
		if ( i < nInputs ) pattern[i] = atof(t);
		else target[i - nInputs] = atof(t);

		//move token onwards
		t = strtok(NULL,",");
		i++;			
	}
	
	//add to records
	data.push_back( new dataEntry( pattern, target ) );		
}

void dataReader::processLine4Image( string &line )
{
	//create new pattern and target
	double* pattern = new double[sImage];
	double* target = new double[nTargets];

	//store inputs
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");

	while ( t!=NULL && i < (sImage + nTargets) )
	{
		if ( i < sImage ) pattern[i] = atof(t);
		else target[i - sImage] = atof(t);

		//move token onwards
		t = strtok(NULL,",");
		i++;
	}

	//add to records
	data.push_back( new dataEntry( pattern, target ) );
}

void dataReader::processLine4maxmin( string &line2 )
{
	//create new pattern and target
	double p_max = 0;
	double p_min = 0;

	//store inputs
	char* cstr = new char[line2.size()+1];
	char* t;
	strcpy(cstr, line2.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");

	while ( t!=NULL && i < 2 )
	{
		if ( i < 1 ) p_min = atof(t);
		else p_max = atof(t);
		//move token onwards
		t = strtok(NULL,",");
		i++;
	}
	//add to records
	ref.push_back( new reference( p_max, p_min ) );
}

/*******************************************************************
* Selects the data set creation approach
********************************************************************/
void dataReader::setCreationApproach( int approach, double param1, double param2 )
{
	//static
	if ( approach == STATIC )
	{
		creationApproach = STATIC;
		
		//only 1 data set
		numTrainingSets = 1;
	}

	//growing
	else if ( approach == GROWING )
	{			
		if ( param1 <= 100.0 && param1 > 0)
		{
			creationApproach = GROWING;
		
			//step size
			growingStepSize = param1 / 100;
			growingLastDataIndex = 0;

			//number of sets
			numTrainingSets = (int) ceil( 1 / growingStepSize );				
		}
	}

	//windowing
	else if ( approach == WINDOWING )
	{
		//if initial size smaller than total entries and step size smaller than set size
		if ( param1 < data.size() && param2 <= param1)
		{
			creationApproach = WINDOWING;
			
			//params
			windowingSetSize = (int) param1;
			windowingStepSize = (int) param2;
			windowingStartIndex = 0;			

			//number of sets
			numTrainingSets = (int) ceil( (double) ( trainingDataEndIndex - windowingSetSize ) / windowingStepSize ) + 1;
		}			
	}

}

/*******************************************************************
* Returns number of data sets created by creation approach
********************************************************************/
int dataReader::getNumTrainingSets()
{
	return numTrainingSets;
}
/*******************************************************************
* Get data set created by creation approach
********************************************************************/
trainingDataSet* dataReader::getTrainingDataSet()
{		
	switch ( creationApproach )
	{	
		case STATIC : createStaticDataSet(); break;
		case GROWING : createGrowingDataSet(); break;
		case WINDOWING : createWindowingDataSet(); break;
	}
	
	return &tSet;
}
/*******************************************************************
* Get all data entries loaded
********************************************************************/
vector<dataEntry*>& dataReader::getAllDataEntries()
{
	return data;
}

/*******************************************************************
* Create a static data set (all the entries)
********************************************************************/
void dataReader::createStaticDataSet()
{
	//training set
	for ( int i = 0; i < trainingDataEndIndex; i++ ) tSet.trainingSet.push_back( data[i] );		
}
/*******************************************************************
* Create a growing data set (contains only a percentage of entries
* and slowly grows till it contains all entries)
********************************************************************/
void dataReader::createGrowingDataSet()
{
	//increase data set by step percentage
	growingLastDataIndex += (int) ceil( growingStepSize * trainingDataEndIndex );		
	if ( growingLastDataIndex > (int) trainingDataEndIndex ) growingLastDataIndex = trainingDataEndIndex;

	//clear sets
	tSet.trainingSet.clear();
	
	//training set
	for ( int i = 0; i < growingLastDataIndex; i++ ) tSet.trainingSet.push_back( data[i] );			
}
/*******************************************************************
* Create a windowed data set ( creates a window over a part of the data
* set and moves it along until it reaches the end of the date set )
********************************************************************/
void dataReader::createWindowingDataSet()
{
	//create end point
	int endIndex = windowingStartIndex + windowingSetSize;
	if ( endIndex > trainingDataEndIndex ) endIndex = trainingDataEndIndex;		

	//clear sets
	tSet.trainingSet.clear();
					
	//training set
	for ( int i = windowingStartIndex; i < endIndex; i++ ) tSet.trainingSet.push_back( data[i] );
			
	//increase start index
	windowingStartIndex += windowingStepSize;
}

///////////////////////////////////////////////////////////////////////////////

Mat
dataReader::Pooling(const Mat &M, int pVert, int pHori, int poolingMethod){
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val = 0.0;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){
                double minVal = 0.0;
                double maxVal = 0.0;
                Point minLoc;
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp.mul(Reciprocal(sumval));
                val = sum(prob.mul(temp))[0];
                prob.release();
            }
            res.ATD(i, j) = val;
            temp.release();
        }
    }
    newM.release();
    return res;
}

double* dataReader::ConvNPooling(double *pattern, int sImage, bool GAP)
{
	int rows = (int)sqrt(sImage);
	Mat img(rows,rows,CV_64FC1);
	for(int j = 0; j < rows; j++) for(int i = 0; i < rows; i++) img.ATD(j, i) = (double)pattern[j*rows+i];

	Mat kk = Mat::ones(6, 6, CV_8UC1);
	erode(img, img, kk);
	normalize(img,img, 0,255,NORM_MINMAX, CV_8UC1);
	img.convertTo(img, CV_64FC1, 1.0/255, 0);

	vector<vector<Mat>> conv(CLayer.size()+1);
	int count=1;
	for (int i = 0; i < CLayer.size()+1; i++) {
		conv[i].reserve(count);
		count = count * CLayer[i].Kernels.size();
	}
	conv[0].push_back(img);
	for(int l=0; l<CLayer.size(); l++){
		// Convolution and Pooling computation
		for(int n = 0; n < conv[l].size(); n++)
		{
			for(int k = 0; k < CLayer[l].Kernels.size(); k++)
			{
				Mat temp = rot90(CLayer[l].Kernels[k], 2);
				Mat tmpconv = convCalc(conv[l][n], temp, CONV_SAME);
				tmpconv = nonLinearity(tmpconv, CLayer[l].non_linear);
				tmpconv = Pooling(tmpconv, CLayer[l].pdim, CLayer[l].pdim, CLayer[l].pmethod);
				conv[l+1].push_back(tmpconv); // conv[l+1].size() == Kernelset.size()
				temp.release(); tmpconv.release();
			}
		}
		if(CLayer[l].LRN==true) 
			for(int k = 0; k < conv[l].size(); k++) normalize(conv[l][k],conv[l][k], 0,1,NORM_MINMAX, CV_64FC1); 
	}
	
	int nPixel = conv[CLayer.size()][0].cols*conv[CLayer.size()][0].rows;
	double* conv_pattern =  new( double[conv[CLayer.size()].size()*nPixel] );
	if (GAP==true) {
		conv_pattern = (new(double[conv[CLayer.size()].size()]));
		for(int k = 0; k < conv[CLayer.size()].size(); k++)
		{
			Scalar meanVal = mean(conv[CLayer.size()][k]);
			double matMean = (double)meanVal.val[0];
			conv_pattern[k] = matMean;
		}
	}
	else {
		conv_pattern = (new(double[conv[CLayer.size()].size()*nPixel]));
		for(int k = 0; k < conv[CLayer.size()].size(); k++)
			for(int j = 0; j < (int)sqrt(nPixel); j++)
				for(int i = 0; i < (int)sqrt(nPixel); i++)
					conv_pattern[k*nPixel+j*conv[CLayer.size()][k].rows+i] = conv[CLayer.size()][k].ATD(j,i);
	}
	for(int k = 0; k < conv.size(); k++) conv[k].clear();
	conv.clear();
	return conv_pattern;
}

double* dataReader::ConvNPooling(Mat pattern, bool GAP)
{
	// Normalization
	pattern.convertTo(pattern, CV_64FC1, 1.0/255, 0);
	vector<vector<Mat>> conv(CLayer.size()+1);
	int count=1;
	for (int i = 0; i < CLayer.size()+1; i++) {
		conv[i].reserve(count);
		count = count * CLayer[i].Kernels.size();
	}
	conv[0].push_back(pattern);
	for(int l=0; l<CLayer.size(); l++){
		// Convolution and Pooling computation
		for(int n = 0; n < conv[l].size(); n++)
		{
			for(int k = 0; k < CLayer[l].Kernels.size(); k++)
			{
				Mat temp = rot90(CLayer[l].Kernels[k], 2);
				Mat tmpconv = convCalc(conv[l][n], temp, CONV_SAME);
				tmpconv = nonLinearity(tmpconv,CLayer[l].non_linear);
				tmpconv = Pooling(tmpconv, CLayer[l].pdim, CLayer[l].pdim, CLayer[l].pmethod);
				conv[l+1].push_back(tmpconv); // conv[l+1].size() == Kernelset.size()
				temp.release(); tmpconv.release();
			}
		}
	}
	int nPixel = conv[CLayer.size()][0].cols*conv[CLayer.size()][0].rows;
	double* conv_pattern =  new( double[conv[CLayer.size()].size()*nPixel] );
	if (GAP==true) {
		conv_pattern = (new(double[conv[CLayer.size()].size()]));
		for(int k = 0; k < conv[CLayer.size()].size(); k++)
		{
			Scalar meanVal = mean(conv[CLayer.size()][k]);
			double matMean = (double)meanVal.val[0];
			conv_pattern[k] = matMean;
		}
	}
	else {
		conv_pattern = (new(double[conv[CLayer.size()].size()*nPixel]));
		for(int k = 0; k < conv[CLayer.size()].size(); k++)
			for(int j = 0; j < (int)sqrt(nPixel); j++)
				for(int i = 0; i < (int)sqrt(nPixel); i++)
					conv_pattern[k*nPixel+j*conv[CLayer.size()][k].rows+i] = conv[CLayer.size()][k].ATD(j,i);
	}
	for(int k = 0; k < conv.size(); k++) conv[k].clear();
	conv.clear();
	return conv_pattern;
}

bool dataReader::loadKernels(const char* filename, vector<ConvLayer> CL)
{
	CLayer.reserve(CL.size());
	CLayer = CL;

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

		//check if sufficient weights were loaded
		int num_weight = 0;
		for(int i=0; i<CLayer.size(); i++) num_weight = num_weight + (int)(CLayer[i].sKernel * CLayer[i].sKernel * CLayer[i].nKernel);
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
			for(int l=0; l<CLayer.size(); l++){
				for ( int k=0; k < CLayer[l].nKernel; k++ )
				{
					Mat buf(CLayer[l].sKernel,CLayer[l].sKernel,CV_64FC1);
					for ( int j=0; j < CLayer[l].sKernel; j++ )
					{
						for ( int i=0; i < CLayer[l].sKernel; i++ ) {
							buf.ATD(j,i) = weights[pos];
							pos++;
						}
					}
					CLayer[l].Kernels.push_back(buf);
					buf.release();
				}
			}
			//print success
			cout << endl << "Convolution Kernels loaded successfully from '" << filename << "'" << endl;
			//close file
			inputFile.close();
			return true;
		}
	}
	else
	{
		cout << endl << "Error - Kernel input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}
