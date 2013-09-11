#include <vector>
//#include <ANN.h>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;



typedef struct point3d {
  double x;
  double y;
  double z;
} _point3d;



point3d make_point(double x, double y, double z) {
  point3d mypoint = {x,y,z};
  return mypoint;
} 
/*
double MAX(double a, double b) {

  if(a>b)
    return a;
  return b;
}
*/
void LoOP_outlier(vector<point3d>& xyz, vector<point3d>& out3d, double lambda, double alpha,int k_search) {
  int kn = k_search;


  point3d threeD;
  //dataPts = annAllocPts(xyz.size(),3);
  vector<int> index(k_search);
  vector<float> dist(k_search);
  cv::flann::KDTreeIndexParams indexParams(5);
  cv::flann::Index kdtree(xyz,indexParams);
  double *expect = new double [kn];
  double *expect2 = new double [kn];
  /*
  int p = 0;
  std::vector<point3d>::iterator it;
  for(it=xyz.begin(); it < xyz.end();it++) {
    threeD = *it;
    dataPts[p][0] = threeD.x;
    dataPts[p][1] = threeD.y;
    dataPts[p++][2] = threeD.z;
  }
  kdTree = new ANNkd_tree (dataPts, (int)xyz.size(),3);
  */
  double lof;
  double LoOP;
  point3d dum;
  //  std::vector<point3d> out3d;
  vector<point3d> q;
  for(int i=0;i<xyz.size();i++)  {
    double x,y,z,kdistA = 0.;

    x  = xyz[i].x;
    y = xyz[i].y;
    z = xyz[i].z; //S;
    q.push_back(make_point(x,y,z));
   
    kdtree.knnSearch(q,index,dist,kn,cv::flann::SearchParams(64));
    q.pop_back();
    for(int k=0;k<kn;k++)
      kdistA += dist[k];
    double S = sqrt(x*x+y*y+z*z);
    double sigm = sqrt(kdistA/S);
    double pdist = lambda * sigm;
    double pdistE = 0.;
    for(int k=0;k<kn;k++) {
      
      x = xyz[index[k]].x;
      y = xyz[index[k]].y;
      z = xyz[index[k]].y;
      q.push_back(make_point(x,y,z));
      kdtree.knnSearch(q,index,dist,kn,cv::flann::SearchParams(64));
      q.pop_back();
      kdistA = 0.;
      for(int n=0;n<kn;n++) 
	kdistA += dist[n];
      double Ss = sqrt(x*x+y*y+z*z);
      kdistA = sqrt(kdistA/Ss);
      expect[k] = lambda*kdistA;
      expect2[k] = expect[k]*expect[k];
    }

    double exAv = 0.;
    double exAv2 =0.;
    for(int k=0;k<kn;k++) {
      exAv += expect[k];
      exAv2 += expect2[k];
    }
    for(int k=0;k<kn;k++) {
      expect[k] = fabs(expect[k]-exAv/kn);
      expect2[k] = fabs(expect2[k]-exAv2/kn);
    }
    exAv =0.;
    exAv2 = 0.;
    for(int k=0;k<kn;k++) {
      exAv += expect[k];
      exAv2 += expect2[k];
    }
    double Ex2 = exAv2/kn;
    pdistE = exAv/kn;
    double plof=pdist/pdistE-1;
    double  nPlof = lambda* sqrt(Ex2*plof*plof);
    double LoOP = MAX(0,erf(plof/1.414*nPlof));
 
    if(LoOP < alpha)
      out3d.push_back(make_point(xyz[i].x,xyz[i].y,xyz[i].z));
  }
  delete [] expect;
  delete [] expect2;
}
  
