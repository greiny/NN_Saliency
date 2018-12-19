#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <pthread.h>
#include <cstdio>
#include <chrono>
#include <unistd.h>
#include <stdlib.h>
#include <ctime>
#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

//custom includes
#include "modules/neuralNetwork.h"
#include "modules/DisplayManyImages.h"
#include "modules/matrix_maths.h"

using namespace std;
using namespace cv;

void makeGaussianPyramid(Mat& src, int s, vector<Mat>& pyr);
vector<Mat> centerSurround(vector<Mat>& fmap1, vector<Mat>& fmap2);
void normalizeMap(vector<Mat>& nmap);

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _USE_MATH_DEFINES
#define pi       3.14159265358979323846

bool Feedforward(Mat mask, Rect roi);
void main_ellipse ();

//Layer Construction
int nInput = 1875;
int nOutput = 3;
vector<int> FCLayer{128}; // #neuron -> FCLayer{256,64}

dataReader dR;
neuralNetwork nn(nInput,FCLayer,nOutput);
float neural_thres = 0.9;
float prob = 0;
int num = 0;

int main(int argc, const char* argv[])
{
	// Loads a csv file of weight matrix data
	char* weights_file = "log/weights4.txt";
	nn.loadWeights(weights_file);
	dR.maxmin("log/maxmin.csv");

	main_ellipse();

    return 0;
}

void main_ellipse ()
{
	vector<cuda::GpuMat> kernels(4);
	for(int k=0; k<kernels.size(); k++) {
		Mat kernel = getGaborKernel(Size(20,20),1, CV_PI/4*k, 30, 0,CV_PI/2);
		kernel.convertTo(kernel,CV_32F);
		kernels[k].upload(kernel);
	}
	int k = 1;
	ofstream logFile;
	logFile.open("data.csv", ios::out);
	if ( logFile.is_open() ) logFile << "Num_img" << "," << "E" << "," << "N_EE"<< endl;

	while(k<1000)
		{
			stringstream img;
			img << "DUT-OMRON/DUT-OMRON-image/img(" << k << ").jpg";
			Mat frame = imread(img.str());
			if (frame.empty() == true) { k++; continue; }
			stringstream gt;
			gt << "DUT-OMRON/DUT-OMRON-GT/img_gt(" << k << ").png";
			Mat frame_gt = imread(gt.str());

			// Step 1.
			vector<Mat> GausnPyr(9);
			makeGaussianPyramid(frame,GausnPyr.size(),GausnPyr);

			// Step 2.
			vector<Mat> Pyr_I(GausnPyr.size()); //Pyr_I[#pyr]
			vector<vector<Mat>> Pyr_C(2); //Pyr_C[#BGR][#pyr]
			vector<vector<Mat>> Pyr_O(4); //Pyr_O[#theta][#pyr]
			for(int i=0; i<GausnPyr.size(); i++) {
				vector<Mat> vtemp(3);
				split(GausnPyr[i], vtemp);

				// Make Intensity Map -> #1
				Pyr_I[i] = (vtemp[0]+vtemp[1]+vtemp[2])/3; //Blue

				// Make Color Map -> #2
				Mat B = vtemp[0]-(vtemp[1]+vtemp[2])/2; //Blue
				Mat Y = (vtemp[2]+vtemp[1])/2-abs((vtemp[2]-vtemp[1])/2)-vtemp[0]; //Yellow
				Mat R = vtemp[2]-(vtemp[1]+vtemp[0])/2; //Red
				Mat G = vtemp[1]-(vtemp[0]+vtemp[2])/2; //Green
				Pyr_C[0].push_back((Mat)(B-Y));
				Pyr_C[1].push_back((Mat)(R-G));
				vtemp.clear();

				// Make Orientation Map -> #4
				Ptr<cuda::Convolution> convolver = cuda::createConvolution(Size(kernels[0].cols,kernels[0].rows));
				cuda::GpuMat buf1(Pyr_I[i]);
				for(int k=0; k<Pyr_O.size(); k++){
					Mat temp;
					cuda::GpuMat buf2;
					buf1.convertTo(buf2,CV_32F);
					cuda::copyMakeBorder(buf2,buf2,kernels[k].cols/2,kernels[k].rows/2,kernels[k].cols/2,kernels[k].rows/2,BORDER_REFLECT_101);
					convolver->convolve(buf2,kernels[k],buf2,true);
					buf2.download(temp);
					Pyr_O[k].push_back(temp);
				}
				buf1.release();
			}
			GausnPyr.clear();

			// Step 3. Center-Surrounded Difference
			vector<Mat> CSD_I,CSD_C,CSD_O;
			CSD_I = centerSurround(Pyr_I,Pyr_I); // 8->6
			Pyr_I.clear();
			for(int k=0; k<Pyr_C.size(); k++) {
				vector<Mat> inv_Pyr_C(Pyr_C[k].size());
				for(int l=0; l<Pyr_C[k].size(); l++) inv_Pyr_C[l] = -Pyr_C[k][l];
				Pyr_C[k] = centerSurround(Pyr_C[k],inv_Pyr_C); //R-G and G-R, B-Y and Y-B
				for(int l=0; l<Pyr_C[k].size(); l++) CSD_C.push_back(Pyr_C[k][l]);
				Pyr_C[k].clear();
			}
			Pyr_C.clear();
			for(int k=0; k<Pyr_O.size(); k++) {
				Pyr_O[k] = centerSurround(Pyr_O[k],Pyr_O[k]);
				for(int l=0; l<Pyr_O[k].size(); l++) CSD_O.push_back(Pyr_O[k][l]);
				Pyr_O[k].clear();
			}
			Pyr_O.clear();

			// Step 4. Normalization
			normalizeMap(CSD_I);
			normalizeMap(CSD_C);
			normalizeMap(CSD_O);

			// Step 5. Conspicuity Maps
			Mat I = Mat::zeros(Size(CSD_I[0].cols,CSD_I[0].rows),CSD_I[0].type());
			Mat C = Mat::zeros(Size(CSD_C[0].cols,CSD_C[0].rows),CSD_C[0].type());
			Mat O = Mat::zeros(Size(CSD_O[0].cols,CSD_O[0].rows),CSD_O[0].type());
			for(int i=0; i<CSD_I.size(); i++) I += CSD_I[i];
			for(int i=0; i<CSD_C.size(); i++) C += CSD_C[i];
			for(int i=0; i<CSD_O.size(); i++) O += CSD_O[i];
			CSD_I.clear(); CSD_C.clear(); CSD_O.clear();

			// Step 6. Merge
			normalize(I,I,0,255,NORM_MINMAX,CV_8UC1);
			normalize(C,C,0,255,NORM_MINMAX,CV_8UC1);
			normalize(O,O,0,255,NORM_MINMAX,CV_8UC1);
			Mat Salmap = (I+C+O)/3;
			Mat Salmap_thres;
			threshold(Salmap, Salmap_thres, 100, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			cvtColor(Salmap_thres,Salmap_thres,COLOR_GRAY2BGR);

			Mat buf_I = I.clone();
			Mat buf_C = C.clone();
			Mat buf_O = O.clone();
			normalize(buf_I,buf_I,0,1,NORM_MINMAX,CV_64F);
			resize(buf_I,buf_I,Size(400,400));
			buf_I = Pooling(buf_I, 8, 8, 0);
			buf_I = Pooling(buf_I, 2, 2, POOL_MEAN);

			normalize(buf_C,buf_C,0,1,NORM_MINMAX,CV_64F);
			resize(buf_C,buf_C,Size(400,400));
			buf_C = Pooling(buf_C, 8, 8, 0);
			buf_C = Pooling(buf_C, 2, 2, POOL_MEAN);

			normalize(buf_O,buf_O,0,1,NORM_MINMAX,CV_64F);
			resize(buf_O,buf_O,Size(400,400));
			buf_O = Pooling(buf_O, 8, 8, 0);
			buf_O = Pooling(buf_O, 2, 2, POOL_MEAN);

			Mat set(Size(buf_O.cols,buf_O.rows*3),buf_O.type());
			vector<Mat> bufbufbuf{buf_I,buf_C,buf_O};
			vconcat(bufbufbuf,set);
			bufbufbuf.clear();
			dR.loadMat4Test(set,nInput,nOutput);
			trainingDataSet *testSet = dR.getTrainingDataSet();;
			double *res = nn.feedForwardPattern(testSet->validationSet[0]->pattern); // calculation results
			float wI = (float)res[0];
			float wC = (float)res[1];
			float wO = (float)res[2];
			float wI2 = (float)(wI/(wI+wC+wO));
			float wC2 = (float)(wC/(wI+wC+wO));
			float wO2 = (float)(wO/(wI+wC+wO));
			Mat NN_Salmap = (wI2*I+wC2*C+wO2*O)/(wI2+wC2+wO2);
			NN_Salmap.convertTo(NN_Salmap,CV_8UC1);
			Mat NN_Salmap_thres;
			threshold(NN_Salmap, NN_Salmap_thres, 100, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			cvtColor(NN_Salmap,NN_Salmap,COLOR_GRAY2BGR);
			cvtColor(NN_Salmap_thres,NN_Salmap_thres,COLOR_GRAY2BGR);
			std::ostringstream ss;
			ss << "Saliency map";
			putText(Salmap_thres, ss.str(), Point(10,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1.2);
			std::ostringstream ss2;
			ss2 << "NN_Saliency map";
			putText(NN_Salmap_thres, ss2.str(), Point(10,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1.2);

			std::ostringstream ww;
			ww.precision(2);
			ww << "wI:"<<wI<<", wC:"<< wC <<", wO:"<<wO;
			putText(NN_Salmap, ww.str(), Point(10,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1.2);

			// Step 7. Display and Save result
			Salmap.convertTo(Salmap,CV_8UC1);
			cvtColor(Salmap,Salmap,COLOR_GRAY2BGR);
			vector<Mat> result={frame,Salmap,NN_Salmap};
			cvtColor(I,I,COLOR_GRAY2BGR);
			cvtColor(C,C,COLOR_GRAY2BGR);
			cvtColor(O,O,COLOR_GRAY2BGR);
			vector<Mat> fmap={frame_gt,Salmap_thres,NN_Salmap_thres};

			float cnt=0;
			float NN_cnt=0;
			for(int h=0; h<NN_Salmap.rows; h++)
				for(int w=0; w<NN_Salmap.cols; w++){
					if (Salmap_thres.data[h*NN_Salmap.cols+w]==frame_gt.data[h*NN_Salmap.cols+w]) cnt++;
					if (NN_Salmap_thres.data[h*NN_Salmap.cols+w]==frame_gt.data[h*NN_Salmap.cols+w]) NN_cnt++;
				}
			logFile << k << "," << cnt/(NN_Salmap.rows*NN_Salmap.cols) << "," << NN_cnt/(NN_Salmap.rows*NN_Salmap.cols)<< endl;

			Mat Rst(Size(frame.cols*3,frame.rows),frame.type(),Scalar::all(0));
			Mat ICO(Size(frame.cols*3,frame.rows),frame.type(),Scalar::all(0));
			Mat dst(Size(frame.cols*3,frame.rows*2),frame.type(),Scalar::all(0));
			hconcat(result, Rst);
			hconcat(fmap, ICO);
			vconcat(Rst, ICO, dst);
			char file_name[255];
			sprintf(file_name,"DUT-result/%d.png",k);
			imwrite(file_name,dst);
			cout << k <<endl;
			k++;

	} // end of while
} // end of main 


void makeGaussianPyramid(Mat& src, int s, vector<Mat>& pyr){
	pyr[0] = src.clone();
	for(int i=1; i<s; i++){
		pyrDown(pyr[i-1],pyr[i], Size((int)(pyr[i-1].cols/2),(int)(pyr[i-1].rows/2)));
		resize(pyr[i],pyr[i],Size(pyr[0].cols,pyr[0].rows));
	}
}

vector<Mat> centerSurround(vector<Mat>& fmap1, vector<Mat>& fmap2){
	vector<int> center = {2,3,4};
	vector<int> delta = {3,4};
	vector<Mat> CSD;
	for(int c=0; c < center.size(); c++)
		for(int d=0; d<delta.size(); d++)
		{
			Mat ctemp = fmap1[center[c]];
			Mat stemp = fmap2[delta[d]+center[c]];
			Mat temp = abs(ctemp-stemp);
			CSD.push_back(temp);
			temp.release();
		}
	return CSD;
}

void normalizeMap(vector<Mat>& nmap){
	for(int i=0; i<nmap.size(); i++)
	{
		normalize(nmap[i],nmap[i],0,1,NORM_MINMAX,CV_32F);
		Scalar meanVal = mean(nmap[i]);
		nmap[i] *= pow((1-(double)meanVal.val[0]),2);
	}
}
