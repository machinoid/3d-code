#ifndef _LOOP_H_
#define _LOOP_H
using namespace std;
typedef struct {
  double x;
  double y;
  double z;
} point3d;

point3d make_point(double x, double y, double z);

void LoOP_outlier(vector<point3d>& xyz, vector<point3d>& out3d, double lambda, double alpha, int k_search);

#endif
