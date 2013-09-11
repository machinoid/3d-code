#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>
#include <iostream>
#include "LoOP.h"
#ifdef _MSC_VER
extern "C" {
#include "XGetopt.h"
}
#else
#include <unistd.h>
#endif
#include "Serial.h"
//#define M_PI CV_PI
using namespace cv;
using namespace std;

void findLaserCenterByRow(const Mat img,cv::vector<Point2f>& laserpts)
{
	static const unsigned char THRESHOLD_MIN_PWR = 25;
	static const unsigned char THRESHOLD_BLOB_DIFF = 10;
	static const int           THRESHOLD_ALLOWED_BLOB_SZ = 20;
	int rowsz = img.cols;
	for(int y=0;y<img.rows;y++) {
		int centerPos = 0;
		unsigned char maxPwr = 0;
		int centerSize = 0;

		int currentPos = 0;
		while (currentPos<rowsz) {
			if (maxPwr<img.at<uchar>(y,currentPos)) {
				centerSize = 1;
				int centerPos_candidate = currentPos;
				unsigned char maxPwr_candidate = img.at<uchar>(y,currentPos);;
				maxPwr=maxPwr_candidate;
				centerPos = centerPos_candidate;
			}
			else 
			{
				++currentPos;
			}
		}

		if (maxPwr < THRESHOLD_MIN_PWR) continue;

		float logicPwr = 0.0f, totalPwr=0.0f;

		for ( currentPos = centerPos-10; currentPos<=centerPos+10; currentPos++)
		{
			float currentPwr;
			if (currentPos>=0 && currentPos<rowsz){
				currentPwr = img.at<uchar>(y,currentPos);
			}else{
				currentPwr = 0.0f;
			}
			logicPwr+=currentPwr;
			totalPwr+=currentPwr*currentPos;
		}
		Point2f pt;
		pt.x = totalPwr/logicPwr;
		pt.y = (float)y;
		laserpts.push_back(pt);
	}
}

int main(int argc,char *argv[]) {
	int thresh = 120;

	double width = 1280.;
	double height = 720.;
	double wall = 53.;
	int ch;
	bool camMdef = false;
	int port = 4;
	double dist = 20.;
	char fsname[255];
	double alpha = 0.1;
	double lambda = 2.;
	int k_search = 15;
	bool doOutliers = false;
	while ((ch = getopt(argc,argv,"h6c:d:a:l:k:o"))!=EOF) {
		switch(ch) {
		case 'h': std::cerr << "usage: pbLaser [-c comport no. -o doOutliers -a outl.alpha -l outl.lambda -k outl.knn -y calibData -d camera-laser distance -w wall dist -6 (640x480)] > output.obj" << endl;
			break;
		case 'd':
			dist = atof(optarg);
			break;
		case 'o': doOutliers = true;
			break;
		case 'a': alpha = atof(optarg);
			break;
		case 'k': k_search = atoi(optarg);
			break;
		case 'l': lambda = atof(optarg);
			break;
		case 'w': wall = atof(optarg);;
			break;
		case 'y': strcpy(fsname,optarg);
			camMdef = true;
			break;
		case '6': width = 640.; height = 480;
			break;
		case 'c': port = atoi(optarg);
			break;
		default: cerr << "invalid option" << endl;
			return -2;
			break;
		}
	}
	Mat cameraIntr;
	Mat cameraDist;
	double fpixx = 649.;
	double fpixy = 649.;
	if(camMdef) {
		FileStorage fs(fsname,FileStorage::READ);
		fs["camera_matrix"] >> cameraIntr;
		fpixx = cameraIntr.at<double>(0,0);
		fpixy = cameraIntr.at<double>(1,1);
		fs["distortion_coefficients"] >> cameraDist;
		fs.release();
	}
	
	CSerial ser;
	if(!ser.Open(port,9600)) {
		std::cerr << "cant open port" << std::endl;
		return -1;
	}

	namedWindow("View",1);
	int start = 0;
	char filen[100];
	
	VideoCapture cap(0);
	if(!cap.isOpened()) {
		cerr << "cannot open camera" << endl;
		return -1;
	}
	if(cap.set(CV_CAP_PROP_FRAME_WIDTH,width))
		cerr << "success width" << endl;
	if(cap.set(CV_CAP_PROP_FRAME_HEIGHT,height))
		cerr << "success height" << endl;

	string inp;
	int p0 = 0;
	Mat dark;
	Mat red[3];
	char buf[100];
	while(1) {
		Mat img;
		cap >> img;
		imshow("View",img);
		int k= waitKey(10);
		if(k=='r') {
		  ser.SendData("-50\n",4);
		  p0 -= 50;
		}
		if(k=='l') {
		  ser.SendData("50\n",3);
		  p0 -= 50;
		}
		if(k=='z')
		  p0 = 0;
		if(k=='s') {
		  sprintf(buf,"%d\n",-p0);
		  ser.SendData(buf,strlen(buf));
		}
		split(img,red);
		dark = red[2].clone();
		//std::cerr << "ROT ";
		//std::cin >> inp;
		if(k=='q')
			break;
		
	}
	cerr << p0 << endl;
	
#if 1
	vector<Point2f> pts;
	vector<point3d> xyz;
	vector<point3d> XYZ_clean;
	point3d xyz3d;
	int nsteps = abs(p0)/4;
	waitKey(-1);
	while(nsteps > 0) {

		
		Mat img;

		cap >> img;
		split(img,red);
		red[2]=red[2]-dark;
		imshow("View",red[2]);
		int k=waitKey(10);
		if(k==27)
			break;
		int orig = 0;
		int yy = 40;
		findLaserCenterByRow(red[2],pts);
		if(pts.size() > 40) {
			Point2f pt;
			for(int i=0;i<pts.size();i++) {
				pt = pts.at(i);
				if(pt.y > 25 && pt.y < 35){
					orig = pt.x;
					break;
				}
			}
			if(orig != 0) {

				double laserAng = atan2((double)orig,fpixx);
				for(int i=0;i<pts.size();i++) {
					pt = pts.at(i);
					double beta = atan2((double)pt.x,fpixx);
					double phi = atan2((double)pt.y,fpixy);
					double d = dist*sin(laserAng)/(sin(laserAng+beta));
					double X0 = d*cos(beta);
					double Z0 = d*sin(beta);
					double Y0 = d*cos(beta)*tan(phi);
					if(doOutliers) {
						xyz3d = make_point(X0,Y0,Z0);
						xyz.push_back(xyz3d);
					} else
						std::cout << "v " << X0 << " " << Y0 << " " << Z0 << std::endl;
				}

			}  else
				cerr << "ORIG ZERO" << endl;
		}
		ser.SendData("-4\n",3);
		nsteps--;
		pts.clear();
	}
	sprintf(buf,"%d\n",abs(p0));
	cerr << buf << endl;

	ser.SendData(buf,strlen(buf));
	if(doOutliers) {
		LoOP_outlier(xyz,XYZ_clean,lambda,alpha,k_search);
		cerr << "cleaned "<< xyz.size()-XYZ_clean.size() << " outliers" << endl;
		vector<point3d>::iterator ito;
		for(ito = XYZ_clean.begin();ito<XYZ_clean.end();ito++) {
			point3d xx;
			xx = *ito;
			cout << "v " << xx.x << " "  << xx.y << " " << xx.z << endl;
		}
	}
	waitKey(-1);

	cap.release();
	return 0;
#endif
}





