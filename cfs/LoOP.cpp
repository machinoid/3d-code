#include <vector>
#include <ANN/ANN.h>
#include <math.h>

using namespace std;


double erf(double);

typedef struct {
  double x;
  double y;
  double z;
} point3d;



point3d make_point(double x, double y, double z) {
  point3d mypoint = {x,y,z};
  return mypoint;
} 

inline double MAX(double a, double b) {

  if(a>b)
    return a;
  return b;
}

void LoOP_outlier(vector<point3d>& xyz, vector<point3d>& out3d, double lambda, double alpha, int k_search) {
  int kn = k_search;
  ANNpointArray dataPts;
  ANNpoint queryPt;
  ANNidxArray nnIdx;
  ANNdistArray dists;
  ANNkd_tree*  kdTree;
  queryPt = annAllocPt(3);
  point3d threeD;
  dataPts = annAllocPts(xyz.size(),3);
  nnIdx = new ANNidx [kn];
  dists = new ANNdist [kn];
  double *expect = new double [kn];
  double *expect2 = new double [kn];
  int p = 0;
  vector<point3d>::iterator it;
  for(it=xyz.begin(); it < xyz.end();it++) {
    threeD = *it;
    dataPts[p][0] = threeD.x;
    dataPts[p][1] = threeD.y;
    dataPts[p++][2] = threeD.z;
  }

  kdTree = new ANNkd_tree (dataPts, xyz.size(),3);
  double lof;
  double LoOP;
  //  std::vector<point3d> out3d;
  for(int i=0;i<xyz.size();i++)  {
    double x,y,z,kdistA = 0.;
    x = queryPt[0] = xyz[i].x;
    y = queryPt[1] = xyz[i].y;
    z = queryPt[2] = xyz[i].z; //S;
    kdTree ->annkSearch(queryPt,kn,nnIdx,dists,1e-3);
    for(int k=0;k<kn;k++)
      kdistA += dists[k];
    double S = sqrt(x*x+y*y+z*z);
    double sigm = sqrt(kdistA/S);
    double pdist = lambda * sigm;
    double pdistE = 0.;
    for(int k=0;k<kn;k++) {
      x = queryPt[0] = xyz[nnIdx[k]].x;
      y = queryPt[1] = xyz[nnIdx[k]].y;
      z = queryPt[2] = xyz[nnIdx[k]].y;
      kdTree->annkSearch(queryPt,kn,nnIdx,dists,1e-3);
      kdistA = 0.;
      for(int n=0;n<kn;n++) 
	kdistA += dists[n];
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
  
