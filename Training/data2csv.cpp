#include <fstream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;

double Reciprocal(const double &);
Mat Reciprocal(const Mat &);
Mat Pooling(const Mat &M, int pVert, int pHori, int poolingMethod);
void makeGaussianPyramid(Mat& src, int s, vector<Mat>& pyr);
vector<Mat> centerSurround(vector<Mat>& fmap1, vector<Mat>& fmap2);
void normalizeMap(vector<Mat>& nmap);
Mat make_depth_histogram(Mat data, int width, int height);

#define ATD at<double>
#define elif else if
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1 //don't use
#define POOL_STOCHASTIC 2

void fliplr(const Mat &, Mat &);
void flipud(const Mat &, Mat &);
void flipudlr(const Mat &, Mat &);
void rotateNScale(const Mat &, Mat &, double, double);
void addWhiteNoise(const Mat &, Mat &, double);
void dataEnlarge(vector<Mat>&, Mat&);

int main (){
	vector<Mat> trainX;
	vector<Mat> trainGT;
	Mat trainY = Mat::zeros(5168, 3, CV_64FC1);
	for ( int i=1 ; i <= 5168 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"DUT-OMRON/DUT-OMRON-image/img(%d).jpg",i);
		buf = imread(file_name);
		trainX.push_back(buf);
	}

	for ( int i=1 ; i <= 5168 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"DUT-OMRON/DUT-OMRON-GT/img_gt(%d).png",i);
		buf = imread(file_name,0);
		normalize(buf,buf, 0,1,NORM_MINMAX,CV_32F);
		trainGT.push_back(buf);
	}

	vector<cuda::GpuMat> kernels(4);
	for(int k=0; k<kernels.size(); k++) {
		Mat kernel = getGaborKernel(Size(20,20),1, CV_PI/4*k, 30, 0,CV_PI/2);
		kernel.convertTo(kernel,CV_32F);
		kernels[k].upload(kernel);
	}

	ofstream imgdata;
	imgdata.open("imgdata.txt",ios::out);
	if ( imgdata.is_open() )
	{

		for ( int i=0 ; i < trainX.size() ; i++ )
		{
			Mat frame = trainX[i];

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

			normalize(I,I,0,1,NORM_MINMAX,CV_32F);
			Mat buf_I = (I & trainGT[i]);
			int sum_I=countNonZero(buf_I);
			resize(I,I,Size(400,400));
			I = Pooling(I, 8, 8, POOL_MAX);
			I = Pooling(I, 2, 2, POOL_MEAN);

			normalize(C,C,0,1,NORM_MINMAX,CV_32F);
			Mat buf_C = (C&trainGT[i]);
			int sum_C=countNonZero(buf_C);
			resize(C,C,Size(400,400));
			C = Pooling(C, 8, 8, POOL_MAX);
			C = Pooling(C, 2, 2, POOL_MEAN);

			normalize(O,O,0,1,NORM_MINMAX,CV_32F);
			Mat buf_O = (O&trainGT[i]);
			int sum_O=countNonZero(buf_O);
			resize(O,O,Size(400,400));
			O = Pooling(O, 8, 8, POOL_MAX);
			O = Pooling(O, 2, 2, POOL_MEAN);

			Mat set(Size(O.cols,O.rows*3),O.type());
			vector<Mat> bufbufbuf{I,C,O};
			vconcat(bufbufbuf,set);
			for(int j = 0; j < set.rows; j++)
				for(int i = 0; i < set.cols; i++)
					imgdata << set.at<double>(j,i) << ",";

			if(sum_I<sum_C && sum_I<sum_O) trainY.ATD(i,0) = 1;
			else if(sum_C<sum_I && sum_C<sum_O) trainY.ATD(i,1) = 1;
			else trainY.ATD(i,2) = 1;
			for(int j = 0; j < trainY.cols; j++) imgdata << trainY.ATD(i, j) << ",";
			imgdata << endl;
		}
	}

	imgdata.close();
	trainGT.clear();

	trainX.clear();
	trainY.release();

	return 0;
}

void
fliplr(const Mat &_from, Mat &_to){
    flip(_from, _to, 1);
}

void
flipud(const Mat &_from, Mat &_to){
    flip(_from, _to, 0);
}

void
flipudlr(const Mat &_from, Mat &_to){
    flip(_from, _to, -1);
}

void
rotateNScale(const Mat &_from, Mat &_to, double angle, double scale){
    Point center = Point(_from.cols / 2, _from.rows / 2);
   // Get the rotation matrix with the specifications above
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
   // Rotate the warped image
    warpAffine(_from, _to, rot_mat, _to.size());
}

void
addWhiteNoise(const Mat &_from, Mat &_to, double stdev){

    _to = Mat::ones(_from.rows, _from.cols, CV_64FC1);
    randu(_to, Scalar(-1.0), Scalar(1.0));
    _to *= stdev;
    _to += _from;
    // how to make this faster?
    for(int i = 0; i < _to.rows; i++){
        for(int j = 0; j < _to.cols; j++){
            if(_to.ATD(i, j) < 0.0) _to.ATD(i, j) = 0.0;
            if(_to.ATD(i, j) > 1.0) _to.ATD(i, j) = 1.0;
        }
    }
}

void 
dataEnlarge(vector<Mat>& data, Mat& label){
    int nSamples = data.size();
  
    /*
    // flip left right
    for(int i = 0; i < nSamples; i++){
        fliplr(data[i], tmp);
        data.push_back(tmp);
    }
    
    // flip left right up down
    for(int i = 0; i < nSamples; i++){
        flipudlr(data[i], tmp);
        data.push_back(tmp);
    }
// add white noise
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        addWhiteNoise(data[i], tmp, 0.05);
        data.push_back(tmp);
    }


*/
    // rotate -10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, -10, 1.2);
        data.push_back(tmp);
    }
    // rotate +10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, 10, 1.2);
        data.push_back(tmp);
    }
    
    // copy label matrix    
    cv::Mat tmp;
    repeat(label, 3, 1, tmp); 
    label = tmp;
}

Mat
Pooling(const Mat &M, int pVert, int pHori, int poolingMethod){
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
                Point minLoc(0,0);
                Point maxLoc(0,0);
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc,noArray() );
                val = maxVal;
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = (double)sum(temp)[0] / (double)(pVert * pHori);
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

double
Reciprocal(const double &s){
    double res = 1.0;
    res /= s;
    return res;
}

Mat
Reciprocal(const Mat &M){
    return 1.0 / M;
}

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

Mat make_depth_histogram(Mat data, int width, int height)
{
	Mat rgb(Size(data.cols,data.rows), CV_8UC3);
    static uint32_t histogram[0x10000];
    memset(histogram, 0, sizeof(histogram));

    for(int j = 0; j < height; ++j) for(int i = 0; i < width; ++i) ++histogram[data.at<uint16_t>(j,i)];
    for(int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i-1]; // Build a cumulative histogram for the indices in [1,0xFFFF]
    for(int j = 0; j < height; ++j)
    	for(int i = 0; i < width; ++i){
			if(uint16_t d = data.at<uint16_t>(j,i))
			{
				int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
				rgb.at<Vec3b>(j, i)[0] = 255 - f;
				rgb.at<Vec3b>(j, i)[1] = 0;
				rgb.at<Vec3b>(j, i)[2] = f;
			}
			else
			{
				rgb.at<Vec3b>(j, i)[0] = 0;
				rgb.at<Vec3b>(j, i)[1] = 5;
				rgb.at<Vec3b>(j, i)[2] = 20;
			}
    	}
    return rgb;
}
