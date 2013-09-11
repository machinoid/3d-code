#include <stdio.h>
//#include <highgui.h>
//#include <cv.h>
#include <fstream>
#include <vector>
//#include <ANN.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#ifdef __GNUC__
#include <unistd.h>
#else
extern "C" {
#include "Xgetopt.h"
}
#endif
#include <stdlib.h>
#define cot(a) (cos((a))/sin((a)))
#if 0
#define DISTANCE 305
#define DISTANCEF 305
#define FRAMEWIDTH 25

#define DISTANCE 485
#define DISTANCEF 485
#define FRAMEWIDTH 21
#endif
int clicked = 0;
int startx=0;
int stopx = 0;
int starty = 0;
//float highLim = DISTANCE *1.;
//float lowLim = DISTANCE * 0.0;

#define SGN(a) (a < 0? -1: 1)

typedef struct {
	float h;
	float s;
	float v;
} hsv_colour;

typedef struct {
	float r;
	float g;
	float b;
} rgb_colour;

struct point3d {
	double x;
	double y;
	double z;
};
typedef struct {
	double x;
	double y;
	double z;
} CvPoint3Df;
void LoOP_outlier(std::vector<point3d>&,std::vector<point3d>&, double, double,int);
point3d make_point(double,double,double);

/**************************************************************
WinFilter version 0.8
http://www.winfilter.20m.com
akundert@hotmail.com

Filter type: Low Pass
Filter model: Butterworth
Filter order: 2
Sampling Frequency: 600 Hz
Cut Frequency: 80.000000 Hz
Coefficents Quantization: float

Z domain Zeros
z = -1.000000 + j 0.000000
z = -1.000000 + j 0.000000

Z domain Poles
z = 0.438635 + j -0.344470
z = 0.438635 + j 0.344470
***************************************************************/
#if 0
#define Ntap 71

float fir(float NewSample) {
	float FIRCoef[Ntap] = { 
		-0.00000000007423267643,
		-0.00000000019445148066,
		-0.00000000031102081599,
		-0.00000000024401585840,
		0.00000000033942653566,
		0.00000000179517315798,
		0.00000000403212667581,
		0.00000000559226794864,
		0.00000000257408062007,
		-0.00000001138204922299,
		-0.00000004150799534275,
		-0.00000008147943606592,
		-0.00000009532019386939,
		-0.00000000021793987214,
		0.00000032201707806107,
		0.00000093326580039679,
		0.00000161119160547108,
		0.00000149757648176227,
		-0.00000114904897853509,
		-0.00000847035301840829,
		-0.00002073848444537501,
		-0.00003134782627461204,
		-0.00001991668297051530,
		0.00005065813292953479,
		0.00021871278635876175,
		0.00046700048559247955,
		0.00060675585671749225,
		0.00013046007877411751,
		-0.00181612137944562070,
		-0.00597189380191800580,
		-0.01137910315480398700,
		-0.01199521888740138300,
		0.00853768574831470330,
		0.08191192665403543300,
		0.25387020556067763000,
		0.37089281332630125000,
		0.25387020556067763000,
		0.08191192665403543300,
		0.00853768574831470330,
		-0.01199521888740138300,
		-0.01137910315480398700,
		-0.00597189380191800580,
		-0.00181612137944562070,
		0.00013046007877411751,
		0.00060675585671749225,
		0.00046700048559247955,
		0.00021871278635876175,
		0.00005065813292953479,
		-0.00001991668297051530,
		-0.00003134782627461204,
		-0.00002073848444537501,
		-0.00000847035301840829,
		-0.00000114904897853509,
		0.00000149757648176227,
		0.00000161119160547108,
		0.00000093326580039679,
		0.00000032201707806107,
		-0.00000000021793987214,
		-0.00000009532019386939,
		-0.00000008147943606592,
		-0.00000004150799534275,
		-0.00000001138204922299,
		0.00000000257408062007,
		0.00000000559226794864,
		0.00000000403212667581,
		0.00000000179517315798,
		0.00000000033942653566,
		-0.00000000024401585840,
		-0.00000000031102081599,
		-0.00000000019445148066,
		-0.00000000007423267643
	};

	static float x[Ntap]; //input samples
	float y=0;            //output sample
	int n;

	//shift the old samples
	for(n=Ntap-1; n>0; n--)
		x[n] = x[n-1];

	//Calculate the new output
	x[0] = NewSample;
	for(n=0; n<Ntap; n++)
		y += FIRCoef[n] * x[n];

	return y;
}
#endif
/**************************************************************
WinFilter version 0.8
http://www.winfilter.20m.com
akundert@hotmail.com

Filter type: Low Pass
Filter model: Butterworth
Filter order: 4
Sampling Frequency: 600 Hz
Cut Frequency: 50.000000 Hz
Coefficents Quantization: float

Z domain Zeros
z = -1.000000 + j 0.000000
z = -1.000000 + j 0.000000
z = -1.000000 + j 0.000000
z = -1.000000 + j 0.000000

Z domain Poles
z = 0.592381 + j -0.130882
z = 0.592381 + j 0.130882
z = 0.726933 + j -0.387747
z = 0.726933 + j 0.387747
***************************************************************/
#define Ntap 31

float fir(float NewSample) {
	float FIRCoef[Ntap] = { 
		0.00350400574222457320,
		0.00461477229191306370,
		0.00465025930374066230,
		0.00295702094413117070,
		-0.00087098231138755589,
		-0.00668123420592172530,
		-0.01344704764415664700,
		-0.01904814407564179400,
		-0.02027727262744168100,
		-0.01321102666985697100,
		0.00594062171612956370,
		0.03945334729509925700,
		0.08622050489525955000,
		0.13956034444883128000,
		0.18497189500859776000,
		0.20332587177695863000,
		0.18497189500859776000,
		0.13956034444883128000,
		0.08622050489525955000,
		0.03945334729509925700,
		0.00594062171612956370,
		-0.01321102666985697100,
		-0.02027727262744168100,
		-0.01904814407564179400,
		-0.01344704764415664700,
		-0.00668123420592172530,
		-0.00087098231138755589,
		0.00295702094413117070,
		0.00465025930374066230,
		0.00461477229191306370,
		0.00350400574222457320
	};

	static float x[Ntap]; //input samples
	float y=0;            //output sample
	int n;

	//shift the old samples
	for(n=Ntap-1; n>0; n--)
		x[n] = x[n-1];

	//Calculate the new output
	x[0] = NewSample;
	for(n=0; n<Ntap; n++)
		y += FIRCoef[n] * x[n];

	return y;
}


void abCoeff(point3d *realPoint, CvPoint *imagePoint, CvMat *result) {
	CvMat *A=cvCreateMat(8,8,CV_64FC1);
	cvZero(A);
	cvmSet(A,0,0, imagePoint[0].x);
	cvmSet(A,0,1,imagePoint[0].y);
	cvmSet(A,0,2,1.);
	cvmSet(A,0,3,-realPoint[0].x*imagePoint[0].x);
	cvmSet(A,0,4,-realPoint[0].x*imagePoint[0].y);
	cvmSet(A,1,0, imagePoint[1].x);
	cvmSet(A,1,1,imagePoint[1].y);
	cvmSet(A,1,2,1.);
	cvmSet(A,1,3,-realPoint[1].x*imagePoint[1].x);
	cvmSet(A,1,4,-realPoint[1].x*imagePoint[1].y);
	cvmSet(A,2,0, imagePoint[2].x);
	cvmSet(A,2,1,imagePoint[2].y);
	cvmSet(A,2,2,1.);
	cvmSet(A,2,3,-realPoint[2].x*imagePoint[2].x);
	cvmSet(A,2,4,-realPoint[2].x*imagePoint[2].y);
	cvmSet(A,3,0, imagePoint[3].x);
	cvmSet(A,3,1,imagePoint[3].y);
	cvmSet(A,3,2,1.);
	cvmSet(A,3,3,-realPoint[3].x*imagePoint[3].x);
	cvmSet(A,3,4,-realPoint[3].x*imagePoint[3].y);

	cvmSet(A,4,3,realPoint[0].y*imagePoint[0].x);
	cvmSet(A,4,4,realPoint[0].y*imagePoint[0].y);
	cvmSet(A,4,5,-imagePoint[0].x);
	cvmSet(A,4,6,-imagePoint[0].y);
	cvmSet(A,4,7,-1.);
	cvmSet(A,5,3, realPoint[1].y*imagePoint[1].x);
	cvmSet(A,5,4,realPoint[1].y*imagePoint[1].y);
	cvmSet(A,5,5,-imagePoint[1].x);
	cvmSet(A,5,6,-imagePoint[1].y);
	cvmSet(A,5,7,-1.);
	cvmSet(A,6,3, realPoint[2].y*imagePoint[2].x);
	cvmSet(A,6,4,realPoint[2].y*imagePoint[2].y);
	cvmSet(A,6,5,-imagePoint[2].x);
	cvmSet(A,5,6,-imagePoint[2].y);
	cvmSet(A,6,7,-1.);
	cvmSet(A,7,3, realPoint[3].y*imagePoint[3].x);
	cvmSet(A,7,4,realPoint[3].y*imagePoint[3].y);
	cvmSet(A,7,5,-imagePoint[3].x);
	cvmSet(A,7,6,-imagePoint[3].y);
	cvmSet(A,7,7,-1.);
	CvMat *b = cvCreateMat(8,1,CV_64FC1);
	cvmSet(b,0,0,realPoint[0].x);
	cvmSet(b,1,0,realPoint[1].x);
	cvmSet(b,2,0,realPoint[2].x);
	cvmSet(b,3,0,realPoint[3].x);
	cvmSet(b,4,0,-realPoint[0].y);
	cvmSet(b,5,0,-realPoint[1].y);
	cvmSet(b,6,0,-realPoint[2].y);
	cvmSet(b,7,0,-realPoint[3].y);
	//  result=cvCreateMat(8,1,CV_32FC1);
	cvSolve(A,b,result,CV_SVD);
	cvReleaseMat(&A);
	cvReleaseMat(&b);
}
int calcCameraCoords(point3d *DB,point3d *DF, point3d *XYZ) {

	double Xi,Xip,Yi,Yip,Zi,Zip;
	double Dx = 0.;
	double Dy = 0.;
	double Dz = 0;
	double c1x=0,c1y=0,c1z=0,b1=0,c2x=0,c2y=0,c2z=0,b2=0,c3x=0,c3y=0,c3z=0,b3=0;
	CvMat *A=cvCreateMat(3,3,CV_64FC1);
	CvMat *X=cvCreateMat(3,1,CV_64FC1);
	CvMat *B=cvCreateMat(3,1,CV_64FC1);
	double Xc=0,Yc=0,Zc=0;
	for(int i=0;i<8;i++) {
		Yi=DB[i].y;
		Yip=DF[i].y;
		Xi=DB[i].x;
		Xip=DF[i].x;
		Zi=DB[i].z;
		Zip=DF[i].z;
		//std::cerr << "Yi " << Yi << " Yip " << Yip << " Zi " << Zi << " Zip " << Zip << std::endl;
		Dx = Xi*Yi - Xi*Yip + Xi*Zi - Xi*Zip - Xip*Yi + Xip*Yip - Xip*Zi + Xip*Zip - pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2);
		if(Dx==0)
			return 1;

		c1x=0;
		c1y += (pow(Xi,2)
			-2*Xi*Xip
			-Xi*Yi
			+Xi*Yip
			+pow(Xip,2)
			+Xip*Yi
			-Xip*Yip
			-Yi*Zi
			+Yi*Zip
			+Yip*Zi
			-Yip*Zip
			+pow(Zi,2)
			-2*Zi*Zip
			+pow(Zip,2))/Dx;
		c1z += (pow(Xi,2)
			-2*Xi*Xip
			-Xi*Zi
			+Xi*Zip
			+pow(Xip,2)
			+Xip*Zi
			-Xip*Zip
			+pow(Yi,2)
			-2*Yi*Yip
			-Yi*Zi
			+Yi*Zip
			+pow(Yip,2)
			+Yip*Zi
			-Yip*Zip)/Dx;


		b1 += -(-pow(Xi,2)*Yip
			-pow(Xi,2)*Zip
			+Xi*Xip*Yi
			+Xi*Xip*Yip
			+Xi*Xip*Zi
			+Xi*Xip*Zip
			-Xi*Yc*Yi
			+Xi*Yi*Yip
			-Xi*pow(Yip,2)
			+Xi*Zi*Zip
			-Xi*pow(Zip,2)
			-pow(Xip,2)*Yi
			-pow(Xip,2)*Zi
			-Xip*pow(Yi,2)
			+Xip*Yi*Yip
			-Xip*pow(Zi,2)
			+Xip*Zi*Zip
			-pow(Yi,2)*Zip
			+Yi*Yip*Zi
			+Yi*Yip*Zip
			+Yi*Zi*Zip
			-Yi*pow(Zip,2)
			-pow(Yip,2)*Zi
			-Yip*pow(Zi,2)
			+Yip*Zi*Zip)/Dx;

		Dy = pow(Xi,2) - 2*Xi*Xip - Xi*Yi + Xi*Yip + pow(Xip,2) + Xip*Yi - Xip*Yip - Yi*Zi + Yi*Zip + Yip*Zi - Yip*Zip + pow(Zi,2) - 2*Zi*Zip + pow(Zip,2);
		if(Dy==0)
			return 1;
		c2x += (Xi*Yi 
			-Xi*Yip 
			+Xi*Zi 
			-Xi*Zip 
			-Xip*Yi 
			+Xip*Yip 
			-Xip*Zi 
			+Xip*Zip 
			-pow(Yi,2) 
			+2*Yi*Yip 
			-pow(Yip,2) 
			-pow(Zi,2) 
			+2*Zi*Zip 
			-pow(Zip,2))/Dy;

		c2y=0;
		c2z += (-pow(Xi,2) 
			+2*Xi*Xip 
			+Xi*Zc*Zi 
			-Xi*Zip 
			-pow(Xip,2) 
			-Xip*Zi 
			+Xip*Zip 
			-pow(Yi,2) 
			+2*Yi*Yip 
			+Yi*Zi 
			-Yi*Zip 
			-pow(Yip,2) 
			-Yip*Zi 
			+Yip*Zip)/Dy;

		b2 += -(+pow(Xi,2)*Yip 
			+pow(Xi,2)*Zip 
			-Xi*Xip*Yi 
			-Xi*Xip*Yip 
			-Xi*Xip*Zi 
			-Xi*Xip*Zip 
			-Xi*Yi*Yip 
			+Xi*pow(Yip,2) 
			-Xi*Zi*Zip 
			+Xi*pow(Zip,2) 
			+pow(Xip,2)*Yi 
			+pow(Xip,2)*Zi 
			+Xip*pow(Yi,2) 
			-Xip*Yi*Yip 
			+Xip*pow(Zi,2)
			-Xip*Zi*Zip 
			+pow(Yi,2)*Zip 
			-Yi*Yip*Zi 
			-Yi*Yip*Zip 
			-Yi*Zi*Zip 
			+Yi*pow(Zip,2) 
			+pow(Yip,2)*Zi 
			+Yip*pow(Zi,2) 
			-Yip*Zi*Zip)/Dy;

		Dz = (Zi - Zip)*(2*Xi - 2*Xip + Yi - Yip);
		if(Dz==0)
			return 1;
		c3x += (-Xi*Yi
			+Xi*Yip
			+Xip*Yi
			-Xip*Yip
			+2*pow(Yi,2)
			-4*Yi*Yip
			+2*pow(Yip,2)
			+2*pow(Zi,2)
			-4*Zi*Zip
			+2*pow(Zip,2))/Dz;
		c3y += (pow(Xi,2)
			-2*Xi*Xip
			-2*Xi*Yi
			+2*Xi*Yip
			+pow(Xip,2)
			+2*Xip*Yi
			-2*Xip*Yip
			+pow(Zi,2)
			-2*Zi*Zip
			+pow(Zip,2))/Dz;

		c3z=0.;
		b3 += -(-pow(Xi,2)*Yip
			+Xi*Xip*Yi
			+Xi*Xip*Yip
			+2*Xi*Yi*Yip
			-2*Xi*pow(Yip,2)
			+2*Xi*Zi*Zip
			-2*Xi*pow(Zip,2)
			-pow(Xip,2)*Yi
			-2*Xip*pow(Yi,2)
			+2*Xip*Yi*Yip
			-2*Xip*pow(Zi,2)
			+2*Xip*Zi*Zip
			+Yi*Zi*Zip
			-Yi*pow(Zip,2)
			-Yip*pow(Zi,2)
			+Yip*Zi*Zip)/Dz;

	}
	cvmSet(A,0,0,c1x);
	cvmSet(A,0,1,c1y);
	cvmSet(A,0,2,c1z);
	cvmSet(A,1,0,c2x);
	cvmSet(A,1,1,c2y);
	cvmSet(A,1,2,c2z);
	cvmSet(A,2,0,c3x);
	cvmSet(A,2,1,c3y);
	cvmSet(A,2,2,c3z);

	cvmSet(B,0,0,b1);
	cvmSet(B,1,0,b2);
	cvmSet(B,2,0,b3);
	cvSolve(A,B,X,CV_SVD);
	Xc = cvmGet(X,0,0);
	Yc = cvmGet(X,1,0);
	Zc = cvmGet(X,2,0);

	XYZ->x = -Xc;
	XYZ->y = -Yc;
	XYZ->z = -Zc;
	return 0;
}
#if 0
Xc=-(Xi*Yip - Xi*r1 - Xip*Yi + Xip*r1)/(Yi - Yip);
Yc=r1;
Zc= -((Yip - r1)*Zi - (Yi - r1)*Zip)/(Yi - Yip);



double r2=-300;
double Xc = 1/2.*((Xi - Xip - 1)*Yi - (Xi - Xip - 1)*Yip - 2*(Xi*Yip - Xi*r2 - Xip*Yi + Xip*r2)*Zi + 2*(Xi*Yip - Xi*r2 - Xip*Yi + Xip*r2)*Zip)/((Yi - Yip)*Zi - (Yi - Yip)*Zip);


double Yc = r2;

double Zc = -1/2.*(2*(Yip - r2)*pow(Zi,3) - 2*(Yi - r2)*pow(Zip,3) - (Yi - Yip)*pow(Zi,2) + (2*(2*Yi + Yip - 3*r2)*Zi - Yi + Yip)*pow(Zip,2) - pow(Yi,3) - 3*Yi*pow(Yip,2) + pow(Yip,3) - (pow(Xi,2) - 2*Xi*Xip + pow(Xip,2))*Yi + (pow(Xi,2) - 2*Xi*Xip + pow(Xip,2) + 3*pow(Yi,2))*Yip - 2*(pow(Xi,2)*r2 + pow(Xip,2)*r2 - (2*Xi*r2 + r2)*Xip + ((2*Xi + 1)*Xip - pow(Xi,2) - pow(Xip,2) - Xi)*Yi + Xi*r2)*Zi - 2*((Yi + 2*Yip - 3*r2)*pow(Zi,2) - pow(Xi,2)*r2 - pow(Xip,2)*r2 - (Yi - Yip)*Zi + (2*Xi*r2 + r2)*Xip - ((2*Xi + 1)*Xip - pow(Xi,2) - pow(Xip,2) - Xi)*Yi - Xi*r2)*Zip)/((Yi - Yip)*pow(Zi,2) - 2*(Yi - Yip)*Zi*Zip + (Yi - Yip)*pow(Zip,2));


double r12 = -(Yi*Yip - pow(Yip,2) + Zi*Zip - pow(Zip,2) + (Yi - Yip)*sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2))))/sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)));

double Y3 = (pow(Yi,2) - Yi*Yip + pow(Zi,2) - Zi*Zip - (Zi - Zip - r12)*sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)))/(Yi - Yip + Zi - Zip + sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)))));

double Z3 = sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2))) + (pow(Yi,2) - Yi*Yip + pow(Zi,2) - Zi*Zip - (Zi - Zip - r12)*sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2))))/(Yi - Yip + Zi - Zip + sqrt(fabs(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2))));

double X3=r12;
XYZ->x=X3;
XYZ->y=Y3;
XYZ->z=Z3;
double r10=300;
double Z2 = -sqrt(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)) + (pow(Yi,2) - Yi*Yip + pow(Zi,2) - Zi*Zip + (Zi - Zip - r10)*sqrt(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)))/(Yi - Yip + Zi - Zip - sqrt(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)));

double Y2 = (pow(Yi,2) - Yi*Yip + pow(Zi,2) - Zi*Zip + (Zi - Zip - r10)*sqrt(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)))/(Yi - Yip + Zi - Zip - sqrt(-pow(Yi,2) + 2*Yi*Yip - pow(Yip,2) - pow(Zi,2) + 2*Zi*Zip - pow(Zip,2)));

double X1 = (Yi*pow(r10,2) + pow(Yip,2)*r10 - (Yi*r10 + pow(r10,2)*Yip + (pow(Yi,2)- 2*Yi*r10 + pow(r10,2))*Zip - ((Yi - r10)*Yip - Yi*r10 + pow(r10,2))*Zi)/(pow(Yi,2) - 2*Yi*Yip + pow(Yip,2)));

XYZ->x = Xc;
XYZ->y = Yc;
XYZ->z = Zc; 
}
#endif
void getPoint(int event, int x, int y, int flags, void* xypoint) {
	CvPoint *xy;
	xy=(CvPoint*)xypoint;
	if(event == CV_EVENT_LBUTTONDOWN) {
		xy->x=x;
		xy->y=y;
		clicked = 1;
	}
}
void xyPoints(double *a, double x, double y, double *X, double *Y){
	*X = (a[0]*x+a[1]*y+a[2])/(a[3]*x+a[4]*y+1);
	*Y = (a[5]*x+a[6]*y+a[7])/(a[3]*x+a[4]*y+1);
}
/*
int calcXYZ(double x, double y, CvMat *coeffFront,CvMat *coeffBack,CvPoint2D32f *frontLeft,CvPoint2D32f *frontRight,CvPoint2D32f *backLeft,CvPoint2D32f* backRight,double *X, double *Y, double *Z) {
	double *a = new double [8];
	double *b= new double [8];
	for(int i=0;i<8;i++) {
		a[i] = cvmGet(coeffFront,i,0);
		b[i] = cvmGet(coeffBack,i,0);
	}
	double xf,yf,zf,xb,yb,zb;
	xyPoints(a,x,y,&xf,&yf);
	xyPoints(b,x,y,&xb,&yb);
	double xz1,xz2,yz1,yz2;
	xyPoints(a,frontLeft->x,frontLeft->y,&xz1,&yz1);
	xyPoints(b,backLeft->x,backLeft->y,&xz2,&yz2);
	double z1,z2;
	double xbz1,ybz1,xbz2,ybz2;
	xyPoints(a,frontRight->x,frontRight->y,&xbz1,&ybz1);
	xyPoints(b,backRight->x,backRight->y,&xbz2,&ybz2);
	z1 = z2 = 0;
	if(yz1-yz2 != 0.)
		z1 = DISTANCE*(yz1-yb)/(yz1-yz2);
	if(ybz1-ybz2 != 0.)
		z2 = DISTANCE*(ybz1-yf)/(ybz1-ybz2);
	if(z1 == 0 && z2 == 0) {
		std::cerr << "NIL" << std::endl;
		return 0;
	}
	*X=(xf+xb)/2.;
	*Y=(yf+yb)/2.;
	//  zf = DISTANCE*(ybz1-yf)/(ybz1-ybz2);//(xbz2-xz1)/DISTANCEF*xf;
	*Z = (z1+z2)/2;
	int retval = 1;
	if(abs(*Z) > DISTANCE) 
		retval = 0;

	delete [] a;
	delete [] b;
	return retval;
}
*/
void bubbleSort(CvPoint *numbers, int array_size)
{
	int i, j;
	CvPoint temp;

	for (i = (array_size - 1); i > 0; i--)
	{
		for (j = 1; j <= i; j++)
		{
			if (numbers[j-1].x > numbers[j].x)
			{
				temp.x = numbers[j-1].x;
				temp.y = numbers[j-1].y;
				numbers[j-1].x = numbers[j].x;
				numbers[j-1].y = numbers[j].y;
				numbers[j].x = temp.x;
				numbers[j].y = temp.y;
			}
		}
	}
}
void  calibrate(point3d *f3d,point3d *b3d,CvMat *coeffFront, CvMat *coeffBack, CvPoint *imgFront,CvPoint *imgBack,IplImage *src, int autom) {
	CvPoint xypoint;
	int pointCount = 0;
	CvPoint *pts=new CvPoint  [8];
	int ptcnt = 0;
	cvNamedWindow("calib",1);
	cvSetMouseCallback("calib",getPoint,(void*)&xypoint);
	while(1) {
		cvShowImage("calib",src);
		cvWaitKey(5);
		if(autom) {

			CvPoint maxp,minp;
			double min,max;
			cvSetImageCOI(src,3);
			cvMinMaxLoc(src,&min,&max,&minp,&maxp,NULL);
			pts[ptcnt].x = maxp.x;
			pts[ptcnt].y = maxp.y;
			//std::cerr << maxp.x << " " << maxp.y << std::endl;
			uchar b =(uchar)src->imageData[maxp.x+maxp.y*src->widthStep];
			uchar g =(uchar)src->imageData[1+maxp.x+maxp.y*src->widthStep];
			uchar r =(uchar)src->imageData[2+maxp.x+maxp.y*src->widthStep];
			//if(r < g || r < b)
			//	continue; 
			cvSetImageCOI(src,0);
			cvCircle(src,maxp,20,CV_RGB(0,0,0),-1);
			cvWaitKey(600);
			ptcnt++;
			if(ptcnt==8) {
				bubbleSort(pts,8);
				if(pts[0].y > pts[1].y) {
					imgFront[0].x = pts[0].x;
					imgFront[0].y= pts[0].y;
					imgFront[1].x = pts[1].x;
					imgFront[1].y= pts[1].y;
				} else {
					imgFront[0].x = pts[1].x;
					imgFront[0].y = pts[1].y;
					imgFront[1].x = pts[0].x;
					imgFront[1].y = pts[0].y;
				}
				if(pts[2].y > pts[3].y) {
					imgBack[0].x = pts[2].x;
					imgBack[0].y= pts[2].y;
					imgBack[1].x = pts[3].x;
					imgBack[1].y= pts[3].y;
				} else {
					imgBack[0].x = pts[3].x;
					imgBack[0].y = pts[3].y;
					imgBack[1].x = pts[2].x;
					imgBack[1].y = pts[2].y;
				}
				if(pts[4].y > pts[5].y) {
					imgBack[3].x = pts[4].x;
					imgBack[3].y= pts[4].y;
					imgBack[2].x = pts[5].x;
					imgBack[2].y= pts[5].y;
				} else {
					imgBack[3].x = pts[5].x;
					imgBack[3].y = pts[5].y;
					imgBack[2].x = pts[4].x;
					imgBack[2].y = pts[4].y;
				}
				if(pts[6].y > pts[7].y) {
					imgFront[3].x = pts[6].x;
					imgFront[3].y= pts[6].y;
					imgFront[2].x = pts[7].x;
					imgFront[2].y= pts[7].y;
				} else {
					imgFront[3].x = pts[7].x;
					imgFront[3].y = pts[7].y;
					imgFront[2].x = pts[6].x;
					imgFront[2].y = pts[6].y;
				}
				//	cvWaitKey(-1);
				//std::cerr << imgFront[0].x << " " << imgFront[0].y << std::endl;
				startx=imgBack[0].x+20;
				stopx = imgBack[2].x-20;
				starty= imgBack[1].y+20;
				break;
			} else
				continue;
		}
		if(clicked) {
			if(pointCount < 8) {
				cvCircle(src,xypoint,4,CV_RGB(255,0,0));
				if(pointCount < 4)
					imgFront[pointCount]=xypoint;
				else if(pointCount < 8)
					imgBack[pointCount-4]=xypoint;
			}
			if(pointCount == 8) {
				cvCircle(src,xypoint,4,CV_RGB(0,255,0));
				startx = xypoint.x;
			}
			if(pointCount == 9) {
				cvCircle(src,xypoint,4,CV_RGB(0,255,0));
				stopx = xypoint.x;
			}
			if(pointCount == 10) {
				cvCircle(src,xypoint,4,CV_RGB(0,0,255));
				starty = xypoint.y;
			}
			if(pointCount > 10)
				break;
			pointCount++;
			clicked = 0;
		}
	} 
	//std::cerr << startx << " " << stopx << " " << starty << std::endl;
	for(int i=0;i<4;i++)
		;//std::cerr << imgBack[i].x << " " << imgFront[i].x << std::endl;
	cvDestroyWindow("calib");
	abCoeff(f3d,imgFront,coeffFront);
	abCoeff(b3d,imgBack,coeffBack);
}


double subpix(IplImage *im, int x, int maxy) {
	double sum=0;
	double fsum=0;
	int p=1;
	if(maxy-3 < 0 || maxy+4 > im->height)
		return cvGetReal2D(im,maxy,x);
	for(int i=maxy-3;i<maxy+4; i++) {
		double pix=cvGetReal2D(im, i,x);
		sum += pix*i;
		fsum += pix;
		p++;
	}
	if(fsum == 0)
		return maxy;
	return sum/fsum;
}

void differentiate(double *deriv, int len) {
	double *tmp = new double [len];
	tmp[0] = 0.;
	for(int i=1;i<len;i++) {
		tmp[i] = deriv[i]-deriv[i-1];
	}
	for(int i=0;i<len;i++) 
		deriv[i] = tmp[i];
	delete [] tmp;
}

double zeroCrossY(double *line, int len, double thresh,int startp, int stop,int plot) {
	std::ofstream plt;
	if(plot)
		plt.open("der.plt");
	int fir_len = len +30;

	double *firS = new double [fir_len];
	double *deriv = new double [fir_len];
	double *fir_line = new double [fir_len];
	for(int i=0;i<fir_len;i++)
		fir_line[i] = 0.;
	for(int i=0;i<len;i++)
		fir_line[i] = line[i];

	for(int i=0;i<fir_len;i++)
		firS[i] = fir(fir_line[i]);
	deriv[0] = 0.;
	double max = 0.;
	int p=0;
	for(int i=1;i<fir_len;i++) {
		deriv[i] = firS[i]-firS[i-1];
		if(firS[i] > max) {
			max = firS[i];
			p=i;
		}
	}
	if(p==fir_len)
		return -1;
	int s1 = p-5;
	int s2 = p+5;
	double zc = 0.;
	for(int k=s1;k<s2;k++) {
		if(SGN(deriv[k]) == 1 && SGN(deriv[k+1]) == -1) {
			zc = k-deriv[k]/(deriv[k+1]-deriv[k])-15.5+startp;
		}
	}
	if(plot) {
		for(int i=0;i<fir_len;i++)
			plt << i << " " << fir_line[i] << " " << firS[i] << " " << deriv[i] << std::endl;
		//std::cerr << max << " " << p << " " << zc << std::endl;
		plt.close();
	}
	if(zc < startp || zc > stop)
		return -1;

	delete [] firS;
	delete [] deriv;
	delete [] fir_line;
	return (zc==0.? -1 : zc);
}
void firS(std::vector<point3d>& data) {
	int len = data.size();
	double *filt = new double [len];
	std::vector<point3d>::iterator it;
	point3d threeD;
	int p=0;
	for(int i=1;i<len;i++) {
		threeD = data[i];
		filt[i] = fir(threeD.z);
	}
	for(int i=0;i<len;i++)  {
		threeD = data[i];
		threeD.z = filt[i];
		data.push_back(threeD);
	}
	delete [] filt;
}

int hsv2rgb(hsv_colour *hsv, rgb_colour *rgb ) {
	/*
	* Purpose:
	* Convert HSV values to RGB values
	* All values are in the range [0.0 .. 1.0]
	*/
	float S, H, V, F, M, N, K;
	int   I;

	S = hsv->s;  /* Saturation */
	H = hsv->h;  /* Hue */
	V = hsv->v;  /* value or brightness */

	if ( S == 0.0 ) {
		/* 
		* Achromatic case, set level of grey 
		*/
		rgb->r = V;
		rgb->g = V;
		rgb->b = V;
	} else {
		/* 
		* Determine levels of primary colours. 
		*/
		if (H >= 1.0) {
			H = 0.0;
		} else {
			H = H * 6;
		} /* end if */
		I = (int) H;   /* should be in the range 0..5 */
		F = H - I;     /* fractional part */

		M = V * (1 - S);
		N = V * (1 - S * F);
		K = V * (1 - S * (1 - F));

		if (I == 0) { rgb->r = V; rgb->g = K; rgb->b = M; }
		if (I == 1) { rgb->r = N; rgb->g = V; rgb->b = M; }
		if (I == 2) { rgb->r = M; rgb->g = V; rgb->b = K; }
		if (I == 3) { rgb->r = M; rgb->g = N; rgb->b = V; }
		if (I == 4) { rgb->r = K; rgb->g = M; rgb->b = V; }
		if (I == 5) { rgb->r = V; rgb->g = M; rgb->b = N; }
	} /* end if */

	return 0;
} /* end function hsv2rgb */
void calcCameraCoords2(CvMat *obj,CvMat* img,point3d *xyz,CvMat* cameraMat, CvMat* distort) {

	CvMat *rvec = cvCreateMat(3,1,CV_32FC1);
	CvMat *tvec = cvCreateMat(3,1,CV_32FC1);
	cvFindExtrinsicCameraParams2(obj,img,cameraMat,distort,rvec,tvec,0);
	xyz->x = cvmGet(tvec,0,0);
	xyz->y = cvmGet(tvec,1,0);
	xyz->z = cvmGet(tvec,2,0);
	//cvReleaseMat(&cameraMat);
	cvReleaseMat(&rvec);
	cvReleaseMat(&tvec);
}

int main(int argc, char** argv)
{
	int ch;
	int kn = 20;
	double lambda = 2.;
	double thresh = 50.;
	double thrM = 1.3;
	int doCalib = 0;
	int doOutliers = 0;
	int noDisplay = 0;
	int perspCorr = 0;
	double alpha = 0.1;
	int autom = 0;
	double zx = 0.2;
	while((ch = getopt(argc,argv,"pACn:l:t:m:i:o1dz:a:")) != EOF) {
		switch(ch) {
		case 'a': alpha = atof(optarg);
			break;
		case 'z': zx = atof(optarg);
			break;
		case 'A': autom=1;
			break;
		case 'p': perspCorr = 1;
			break;
		case 'd': noDisplay = 1;
			break;
		case 'C': doCalib = 1;
			break;
		case 'n': kn = atoi(optarg);
			break;
		case 'l': lambda = atof(optarg);
			break;
		case 't':  thresh = atof(optarg);
			break;
		case 'o': doOutliers = 1;
			break;
		}
	}
	CvMat *coeffFront = cvCreateMat(8,1,CV_64FC1);
	CvMat *coeffBack = cvCreateMat(8,1,CV_64FC1);
	point3d *f3d = new point3d [4];
	point3d *b3d = new point3d [4];
	char filename[300];
	CvMat *imgPt = cvCreateMat(8,2,CV_32FC1);
	CvMat *objPt = cvCreateMat(8,3,CV_32FC1);
#if 0
	// front frame world coordinates
	f3d[0].x =0;
	f3d[0].y=0;
	f3d[0].z=0;
	f3d[1].x=0;
	f3d[1].y=245;
	f3d[1].z=21;
	f3d[2].x=374;
	f3d[2].y=246;
	f3d[2].z=15;
	f3d[3].x=382;
	f3d[3].y=0;
	f3d[3].z=0;
	// back frame world coordinates
	b3d[0].x =72;
	b3d[0].y=0;
	b3d[0].z = 303;

	b3d[1].x =72;
	b3d[1].y=250;
	b3d[1].z=293;

	b3d[2].x= 300;
	b3d[2].y=250;
	b3d[2].z = 303;
	b3d[3].x=309;
	b3d[3].y=0;
	b3d[3].z=300;

	f3d[0].x=0;
	f3d[0].y=0;
	f3d[0].z=0;
	f3d[1].x=0;
	f3d[1].y=410;
	f3d[1].z=0;
	f3d[2].x = 455;
	f3d[2].z=0;
	f3d[2].y=410;
	f3d[3].x=455;
	f3d[3].y=0;
	f3d[3].z=0;
	b3d[0].x=104;
	b3d[0].y=0;
	b3d[1].x=104;
	b3d[1].y=460;
	b3d[2].x=249;
	b3d[2].y=460;
	b3d[3].x=455;
	b3d[3].y=0;
	b3d[0].z=485;
	b3d[1].z=485;
	b3d[2].z=485;
	b3d[3].z=485;
#endif
	
	std::ifstream coord("coord3d.dat");
	coord >> f3d[0].x >> f3d[0].y >> f3d[0].z; 
	coord >> f3d[1].x >> f3d[1].y >> f3d[1].z; 
	coord >> f3d[2].x >> f3d[2].y >> f3d[2].z; 
	coord >> f3d[3].x >> f3d[3].y >> f3d[3].z;
	coord >> b3d[0].x >> b3d[0].y >> b3d[0].z;
	coord >> b3d[1].x >> b3d[1].y >> b3d[1].z;
	coord >> b3d[2].x >> b3d[2].y >> b3d[2].z;
	coord >> b3d[3].x >> b3d[3].y >> b3d[3].z;
	coord.close();
	
	double zLim = zx*b3d[0].z;
	CvFileStorage* fs = cvOpenFileStorage("mslv5000.yml",NULL,CV_STORAGE_READ);
	CvMat* CamMat = (CvMat*)cvReadByName(fs,NULL,"camera_matrix");
	CvMat* distort = (CvMat*)cvReadByName(fs,NULL,"distortion_coefficients");
	CvPoint *imgFront = new CvPoint [4];
	CvPoint *imgBack =  new CvPoint [4];
	if(noDisplay == 1) {
		cvNamedWindow( "Source", 1 );
		cvNamedWindow( "Diff", 1 );
	}
	CvCapture *cap;
	int cam = 0;
	if(strcmp(argv[optind],"cam") == 0) {
		cap = cvCreateCameraCapture(-1);
		cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH,640);
		cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT,480);
		cam = 1;
	} else
		cap = cvCreateFileCapture(argv[optind]);
	if(cap == NULL) {
		fprintf(stderr,"unable to create capture\n");
		return -1;
	} 
	IplImage *src=cvQueryFrame(cap);

	if(src==NULL) {
		fprintf(stderr,"Null frame, exiting..\n");
		return -1;
	}
	IplImage *diff=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	IplImage *dark = cvCreateImage(cvGetSize(src),8,1);
	CvSize sz = cvGetSize(src);
	CvPoint2D32f frontLeft;
	CvPoint2D32f frontRight;
	CvPoint2D32f backLeft;
	CvPoint2D32f backRight;
	point3d XcYcZc;
	CvPoint2D32f *srcp = new CvPoint2D32f [4];
	CvPoint2D32f *dstp = new CvPoint2D32f [4];
	CvPoint* point = new CvPoint [9];
	double *testline = new double [128];
	std::vector<point3d> xyz;
	point3d threeD;
	int k;
	int plot = 1;
	IplImage *und = cvCreateImage(cvGetSize(src),8,3);
	std::ofstream cd;
	//CvVideoWriter *vw = cvCreateVideoWriter("video1.avi",-1,30,cvSize(640,480),1); 
	if(doCalib) {
		std::cerr << " press d to start calibrating, q to quit " << std::endl;
		while(1) {
			src = cvQueryFrame(cap);
			cvUndistort2(src,und,CamMat,distort);

			cvShowImage("Source",und);
			if(cam) 
				k = cvWaitKey(5);
			else
				k = cvWaitKey(-1);
			if(k=='q')
				return 1;
			//cvWriteFrame(vw,src);
			if(k=='d')
				break;
		}
		//cvReleaseVideoWriter(&vw);
		// exit(0); return 0;
		calibrate(f3d,b3d,coeffFront, coeffBack, imgFront,imgBack,src,autom);
		for(int i=0;i<4;i++)
			;//std::cerr  << "f x " << imgFront[i].x << " " << imgFront[i].y << std::endl;

		cd.open("calib.dat");

		cd << imgFront[0].x << " " << imgFront[0].y << std::endl;
		cd << imgFront[1].x << " " << imgFront[1].y << std::endl;
		cd << imgFront[2].x << " " << imgFront[2].y << std::endl;
		cd  << imgFront[3].x << " " << imgFront[3].y << std::endl;
		cd << imgBack[0].x << " " << imgBack[0].y << std::endl;
		cd << imgBack[1].x << " " << imgBack[1].y << std::endl;
		cd << imgBack[2].x << " " << imgBack[2].y << std::endl;
		cd << imgBack[3].x << " " << imgBack[3].y << std::endl;
		cd << startx << " " << stopx << " " <<  starty << std::endl;;
		for(int i=0;i<8;i++)
			cd << cvmGet(coeffFront,i,0) << " ";
		cd << std::endl;
		for(int i=0;i<8;i++)
		 cd  << cvmGet(coeffBack,i,0) << " " ;
		cd << std::endl;
		//cd.close(); 
	}   
	else {
		std::ifstream cd;
		cd.open("calib.dat");
		cd >>imgFront[0].x >>imgFront[0].y;
		cd >>imgFront[1].x>>imgFront[1].y;
		cd >>imgFront[2].x >>imgFront[2].y;
		cd >>imgFront[3].x >>imgFront[3].y;
		cd >>imgBack[0].x >>imgBack[0].y;
		cd >>imgBack[1].x >>imgBack[1].y;
		cd >>imgBack[2].x >>imgBack[2].y;
		cd >>imgBack[3].x >>imgBack[3].y;
		cd >>startx >>stopx >>starty;
		for(int i=0;i<8;i++) {
			double d;
			cd >> d;
			cvmSet(coeffFront,i,0,d);
		}
		for(int i=0;i<8;i++) {
			double d;
			cd >> d;
			cvmSet(coeffBack,i,0,d);
		}
		cd >> XcYcZc.x >> XcYcZc.y >> XcYcZc.z;
		cd.close();
	}


	point3d l1,l2,l3,l4;
	int cameraNotDone = 1;
	int cp = 0;
	int doIt = 1;
	double *line = new double [src->height];
	int cnt = 0;

	cvSplit(und,NULL,NULL,dark,NULL);
	double mini,maxi;
	double *a = new double [8];
	double *b= new double [8];

	for(int i=0;i<8;i++) {
		a[i] = cvmGet(coeffFront,i,0);
		b[i] = cvmGet(coeffBack,i,0);
	}
	//cvShowImage("Source",warp);
	//cvWaitKey(-1);
	std::vector<point3d>::iterator it;
	IplImage *camImg = cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,3);
	CvMat *matZ = cvCreateMat(4,1,CV_32FC1);
	CvMat *matX = cvCreateMat(4,3,CV_32FC1);
	CvMat *matRes = cvCreateMat(3,1,CV_32FC1);
	cvZero(camImg);
	if(cam) {
		std::cerr << " adjust camera bright. etc. Don't move camera! d to start" << std::endl;
		while(1) {

			src = cvQueryFrame(cap);
			cvUndistort2(src,und,CamMat,distort);
			cvShowImage("Source",und);
			int k=cvWaitKey(5);
			if(k=='d')
				break;
		}
	}
	std::ofstream abc("abc.obj");

	while(doIt) {
		src=cvQueryFrame(cap);
		cvUndistort2(src,und,CamMat,distort);
		if(src == NULL) {
			fprintf(stderr,"EOF\n");
			break;
		}

		//cvSmooth(und,und,CV_MEDIAN,5);
		cvSplit(und,NULL,NULL,diff,NULL);

		cvSub(diff,dark,diff);
		cvMinMaxLoc(diff,&mini,&maxi,NULL,NULL);

		if(maxi < thresh)
			continue;
		k=cvWaitKey(5);
		if(k=='q')
			break;
		cvShowImage("Source",diff);
		int pt = 0;
		double max = 0.;
		for(int i=imgFront[1].y+10;i< imgFront[0].y-10;i++) {
			double pix = cvGetReal2D(diff,i,imgFront[0].x);
			if(pix > max) {
				max = pix;
				pt = i;
			}
		}
		if(pt == 0)
			continue;
		if(max < thresh)
			continue;
		frontLeft.x = imgFront[0].x;
		frontLeft.y = subpix(diff,frontLeft.x,pt);
		if(fabs(frontLeft.y-pt) > 5)
			continue;
		pt = 0;
		max =0.;
		for(int i=imgBack[1].y+10;i<imgBack[0].y-10;i++) {
			double pix = cvGetReal2D(diff,i,imgBack[0].x);
			if(pix > max) {
				max = pix;
				pt = i;
			}
		}
		if(pt == 0)
			continue;
		if(max < thresh)
			continue;
		backLeft.x = imgBack[0].x;
		backLeft.y = subpix(diff,backLeft.x,pt);
		if(fabs(backLeft.y-pt)> 5)
			continue;
		pt = 0;
		max =0.;
		for(int i=imgBack[2].y+10;i<imgBack[3].y-10;i++) {
			double pix = cvGetReal2D(diff,i,imgBack[2].x);
			if(pix > max) {
				max = pix;
				pt = i;
			}
		}
		//std::cerr << pt << " " << max << std::endl;
		if(pt == 0)
			continue;
		if(max < thresh)
			continue;
		backRight.x = imgBack[0].x;
		backRight.y = subpix(diff,backRight.x,pt);
		if(fabs(backRight.y-pt) > 5)
			continue;
		pt = 0;
		max = 0.;
		for(int i=imgFront[2].y+10;i<imgBack[3].y-10;i++) {
			double pix = cvGetReal2D(diff,i,imgFront[3].x);
			if(pix > max) {
				max = pix;
				pt = i;
			}
		}
		if(pt == 0)
			continue;
		if(max < thresh)
			continue;
		frontRight.x = imgBack[3].x;
		frontRight.y = subpix(diff,frontRight.x,pt);
		if(fabs(frontRight.y-pt) > 5)
			continue;
		xyPoints(a,frontLeft.x,frontLeft.y,&l1.x,&l1.y);
		double z1 = f3d[2].z*frontLeft.y/(imgFront[0].y-imgFront[1].y); 
		l1.z = z1;;
		xyPoints(a,frontRight.x,frontRight.y,&l2.x,&l2.y);
		double z2 = f3d[3].z*frontRight.y/(imgFront[3].y-imgFront[2].y); 
		l2.z = z2;
		xyPoints(b,backLeft.x,backLeft.y,&l3.x,&l3.y);
		l3.z = b3d[0].z;
		xyPoints(b,backRight.x,backRight.y,&l4.x,&l4.y);
		l4.z = z1;//f3d[0].z;
		l4.z =b3d[2].z;
		std::cerr << l1.x << " " << l1.y << " " << l2.y << " " << l4.z << std::endl;
		if(doCalib && cameraNotDone ) {
			cvmSet(objPt,cp,0,f3d[0].x);
			cvmSet(objPt,cp,1,f3d[0].y);
			cvmSet(objPt,cp,2,f3d[0].z);
			cp++;
			cvmSet(objPt,cp,0,f3d[1].x);
			cvmSet(objPt,cp,1,f3d[1].y);
			cvmSet(objPt,cp,2,f3d[1].z);
			cp++;
			cvmSet(objPt,cp,0,f3d[2].x);
			cvmSet(objPt,cp,1,f3d[2].y);
			cvmSet(objPt,cp,2,f3d[2].z);
			cp++;
			cvmSet(objPt,cp,0,f3d[3].x);
			cvmSet(objPt,cp,1,f3d[3].y);
			cvmSet(objPt,cp,2,f3d[3].z);
			cp++;
			cvmSet(objPt,cp,0,b3d[0].x);
			cvmSet(objPt,cp,1,b3d[0].y);
			cvmSet(objPt,cp,2,b3d[0].z);
			cp++;
			cvmSet(objPt,cp,0,b3d[1].x);
			cvmSet(objPt,cp,1,b3d[1].y);
			cvmSet(objPt,cp,2,b3d[1].z);
			cp++;
			cvmSet(objPt,cp,0,b3d[2].x);
			cvmSet(objPt,cp,1,b3d[2].y);
			cvmSet(objPt,cp,2,b3d[2].z);
			cp++;
			cvmSet(objPt,cp,0,b3d[3].x);
			cvmSet(objPt,cp,1,b3d[3].y);
			cvmSet(objPt,cp,1,b3d[3].z);
			cp++;
			cvmSet(imgPt,cp-8,0,imgFront[0].x);
			cvmSet(imgPt,cp-8,1,imgFront[0].y);
			cvmSet(imgPt,cp-7,0,imgFront[1].x);
			cvmSet(imgPt,cp-7,1,imgFront[1].y);
			cvmSet(imgPt,cp-6,0,imgFront[2].x);
			cvmSet(imgPt,cp-6,1,imgFront[2].y);
			cvmSet(imgPt,cp-5,0,imgFront[3].x);
			cvmSet(imgPt,cp-5,1,imgFront[3].y);
			cvmSet(imgPt,cp-4,0,imgBack[0].x);
			cvmSet(imgPt,cp-4,1,imgBack[0].y);
			cvmSet(imgPt,cp-3,0,imgBack[1].x);
			cvmSet(imgPt,cp-3,1,imgBack[1].y);
			cvmSet(imgPt,cp-2,0,imgBack[2].x);
			cvmSet(imgPt,cp-2,1,imgBack[2].y);
			cvmSet(imgPt,cp-1,0,imgBack[3].x);
			cvmSet(imgPt,cp-1,1,imgBack[3].y);
			calcCameraCoords2(objPt,imgPt,&XcYcZc,CamMat,distort);
			std::cerr << "camera coords " << XcYcZc.x << " " << XcYcZc.y << " " << XcYcZc.z << std::endl;
			//cd << XcYcZc.x << " " << XcYcZc.y << " " << XcYcZc.z << std::endl;
			//cd.close();
			cameraNotDone=0;
		}
		max = 0.;
		for(int j=startx;j<stopx;j++) {
			int p = 0;
			for(int i=starty;i<src->height;i++) {
				double pix = cvGetReal2D(diff,i,j);
				if(pix > max) {
					max = pix;
					p=i;
				}
				//line[p++] = pix;
			}
			if(p<starty)
				continue;
			//double zc = zeroCrossY(line,src->height,thresh,starty,src->height,0);
			double zc = subpix(diff,j,p);
			//std::cerr << zc << std::endl;
			double X,Y;
			xyPoints(b,j,zc,&X,&Y);
			//std::cerr << " X " << X << " Y " << Y << std::endl;
			cvmSet(matZ,0,0,l1.z);
			cvmSet(matZ,1,0,l2.z);
			cvmSet(matZ,2,0,l3.z);
			cvmSet(matZ,3,0,l4.z);
			cvmSet(matX,0,0,l1.x);
			cvmSet(matX,1,0,l2.x);
			cvmSet(matX,2,0,l3.x);
			cvmSet(matX,3,0,l4.x);
			cvmSet(matX,0,1,l1.y);
			cvmSet(matX,1,1,l2.y);
			cvmSet(matX,2,1,l3.y);
			cvmSet(matX,3,1,l4.y);
			cvmSet(matX,0,2,1.);
			cvmSet(matX,1,2,1.);
			cvmSet(matX,2,2,1.);
			cvmSet(matX,3,2,1.);
			cvSolve(matX,matZ,matRes,CV_SVD);
			double A = cvmGet(matRes,0,0);
			double B = cvmGet(matRes,1,0);
			double C = cvmGet(matRes,2,0);
			double Z1 = A*X+B*Y+C;
			double t;
			double x0 = (XcYcZc.x);
			double y0 = (XcYcZc.y);
			double z0 = (XcYcZc.z);
			//std::cerr << "A " << A << " B " << B << " C " << C << " x0 " << x0 << std::endl;
			// AX+BY+C-Z=0;
			// t=-C/(A*x0+B*y0-z0);
			// t=-(A*X+B*Y-DISTANCE+C)/(A*x0+B*y0-z0);
			t=(A*x0+B*y0-z0+C)/(A*(x0-X)+B*(y0-Y)-(z0-b3d[1].z));
			//std::cerr << "t " << t << std::endl;

			if(t>0 && t<1.)
				;//std::cerr << "positive t " << t << std::endl;
			//	continue;
			else {
				//std::cerr << "negative t " << t << std::endl;
				continue;
			}

			double XX=x0+(X-x0)*t;
			double YY=y0+(Y-y0)*t;
			double ZZ=z0+(b3d[1].z-z0)*t;
			if(YY < 0)
				continue;
			if(ZZ <0)
				continue;
			abc << "v " << XX << " " << YY << " " << ZZ << std::endl;
			if(ZZ > zLim)
				xyz.push_back(make_point(XX,YY,ZZ));
			else
				continue;
			rgb_colour rgb;
			hsv_colour hsv;
			hsv.s = 0.8;
			hsv.v = 0.8;
			hsv.h =(ZZ/b3d[0].z);
			hsv2rgb(&hsv,&rgb);
			CvPoint  colour;
			colour.x = j;
			colour.y = (int)zc;
			cvCircle(camImg,colour,1,CV_RGB((int)(rgb.r*255.),(int)(rgb.g*255),(int)(rgb.b*255.)));
			cvShowImage("Diff",camImg);


		}
		if(cvGetCaptureProperty(cap,CV_CAP_PROP_POS_AVI_RATIO) > 0.9)
			doIt = 0;
	}
	abc.close();
	cvDestroyWindow("Source");
	cvDestroyWindow("Diff");
	cvWaitKey(4);
	cvReleaseMat(&matZ);
	cvReleaseMat(&matX);
	cvReleaseMat(&matRes);
	if(doOutliers == 0) {
		for(it=xyz.begin();it < xyz.end();it++) {
			threeD = *it;
			std::cout << "v " << threeD.x << " " << threeD.y << " " << threeD.z << std::endl;
		}
		cvReleaseCapture(&cap);
		return 0;
	}
	std::vector<point3d> out3d;
	LoOP_outlier(xyz,out3d, lambda,alpha, kn);

	std::vector<point3d>::iterator ito;
	for(ito = out3d.begin();ito<out3d.end();ito++) {
		threeD = *ito;
		std::cout << "v " << threeD.x << " " << threeD.y << " " <<threeD.z << std::endl;
	}
	std::cerr << "Total # of vertices " << xyz.size() << ", removed " << xyz.size()-out3d.size() << " outliers" << std::endl;
	std::cerr << "done" << std::endl;

	cvReleaseCapture(&cap);
	return 0;

}

