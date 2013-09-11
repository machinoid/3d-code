
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "projector.h"
#ifndef __GNUC__
extern "C" {
#include "XGetopt.h"
}
#else
#include <unistd.h>
#endif

#define LOOKAT
//#define M_PI 3.1415926
using namespace cv;
using namespace std;
bool changed = false;
bool noiseChanged = false;

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

class WrappedPixel {
public:
  int x,y;
  float dist,phase;
  WrappedPixel(int x, int y, float dist, float phase) {
    this.x = x;
    this.y = y;
    this.dist = dist;
    this.phase = phase;
  }
  int compareTo(WrappedPixel *pix) {
    if(pix->dist == dist)
      return 0;
    if(pix->dist < dist)
      return -1;
    else
      return 0;
  }
};
class Phase {
  Mat process;
  vector<WrappedPixel> toProcess;
  Mat phase,zImg;
  Mat mask;
  Mat im1,im2,im3;
  Mat inpMask;
  int height,width,rd;
  float zskew,zscale,noiseThr;
  float contrast_;

public:
  float contrast(int pix) {
    float f =(259*(contrast_+255)) / (255*(259-contrast_));
    float cpix = f*(pix-128)+128;
    return cpix;
  }
  void phaseUnwrap(float basePhase, int x,int y) {
    if(process.at<uchar>(y,x) != 0) {
      float diff = phase.at<float>(y,x) -(basePhase - (int)basePhase);
      if(diff > 0.5f)
	diff -= 1;
      if(diff < -0.5f)
	diff += 1.f;
      phase.at<float>(y,x) = basePhase+diff;
      process.at<uchar>(y,x) = 0;
      WrappedPixel xy = new WrappedPixel(;
      xy.x = x;
      xy.y = y;
      toProcess.push_back(xy);
    }
  }

  void phaseUnwrap() {
    WrappedPixel xy = new WrappedPixel(width/2,height/2,0,phase.at<float>(height/2,width/2));
    
    toProcess.push_back(xy);
    int c = 0;
    while(!toProcess.empty()) {
      //cerr << toProcess.size() << endl;
      xy = toProcess.front();
      toProcess.erase(toProcess.begin());
      int x = xy.x;
      int y = xy.y;
      float r = phase.at<float>(y,x);
      if(y>0)
	phaseUnwrap(r,x,y-1);
      if(y < height-1)
	phaseUnwrap(r,x,y+1);
      if(x>0)
	phaseUnwrap(r,x-1,y);
      if(x<width-1)
	phaseUnwrap(r,x+1,y);
      c++;
    }
    //cerr << "unwrap end" << " " << toProcess.size() << " cnt " << c <<  endl;
  }
  void loadImage(String i1name,String i2name,String i3name) {
    im1 = imread(i1name,0);
    if(im1.empty())
      exit(-1);
    im2 = imread(i2name,0);
    if(im2.empty())
      exit(-1);
    im3 = imread(i3name,0);
    if(im3.empty())
      exit(-1);
    height = im3.rows;
    width = im3.cols;
    phase = Mat::zeros(im3.size(),CV_32F);
    zImg = Mat::zeros(im3.size(),CV_8UC3);

  }
  void phaseWrap() {
    float sqrt3 = sqrt(3.);
    for(int y=0;y<height;y++)
      for(int x=0;x<width;x++) {
	uchar c1 = im1.at<uchar>(y,x);
	uchar c2 = im2.at<uchar>(y,x);
	uchar c3 = im3.at<uchar>(y,x);
	float p1 = (contrast(c1)/255.f);
	float p2 = (contrast(c2)/255.f);
	float p3 = (contrast(c3)/255.f);
	float psum = p1+p2+p3;
	if(psum == 0.) {
	  //cerr << "psum zero" << endl;
	  continue;
	}
	float pRange = MAX(MAX(p1,p2),p3)-MIN(MIN(p1,p2),p3);
	float gamma = pRange/psum;
	mask.at<uchar>(y,x) = gamma < noiseThr;
	//cerr << countNonZero(mask) << " gamma " << gamma << " noise " << noiseThr << endl;
	process.at<uchar>(y,x) = (mask.at<uchar>(y,x) == 0 ? 1 : 0);
	phase.at<float>(y,x) = atan2(sqrt3*(p1-p3),2*p2-p1-p3)/(2*CV_PI);
      }
    //cerr << "pn " << countNonZero(process) << endl;

  }
  void setScale(float skew,float scale,float thr,float contr) {
    noiseThr = thr;
    zskew = skew;
    zscale = scale;
    contrast_ = contr;
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

  void getDist(Mat& dist) {
    dist = Mat::zeros(im1.size(),CV_32F);
    hsv_colour hsv;
    rgb_colour rgb;
    for(int y=0;y<height;y+= rd) {
      float planephase = 0.5f-(y-(height/2.f))/zskew;
      for(int x=0;x<width;x += rd) {

	if(mask.at<uchar>(y,x)==0 && inpMask.at<uchar>(y,x) == 0) 
	  dist.at<float>(y,x) = (phase.at<float>(y,x)-planephase)*zscale;
      }
    }
    /*
      double min,max;
      minMaxLoc(dist,&min,&max);
      Mat show;
      show = 1./(max-min)*(dist-min);
      for(int y=0;y<height;y+=rd)
      for(int x=0;x<width;x+=rd) {
      if(mask.at<uchar>(y,x) == 0) {
      hsv.s = 0.8;
      hsv.v = 0.8;
      hsv.h = show.at<float>(y,x);
      hsv2rgb(&hsv,&rgb);
      Point col;
      col.x = x;
      col.y = y;
      circle(zImg,col,1,CV_RGB((int)(rgb.r*255.),(int)(rgb.g*255.),(int)(rgb.b*255.)));
      imshow("Z",zImg);
      }
      }
      waitKey(1);
    */
  }

  void clearImg() {
    zImg = Scalar::all(0);
  }

  Point3f center(const Mat& xyz) {

    Scalar m=mean(xyz);
    Point3f c(m[0],m[1],m[2]);
    return c;
  }

  void create3d(string fn,const Mat& points,const Mat& color) {
    ofstream obj(fn.c_str());
    ofstream vxl("voxel.vxl");
    Mat x(points.size(),CV_32F);
    Mat y(points.size(),CV_32F);
    Mat z(points.size(),CV_32F);
    Mat r(points.size(),CV_8UC1);
    Mat g(points.size(),CV_8UC1);
    Mat b(points.size(),CV_8UC1);
    Mat xyz[3] = {x,y,z};
    split(points,xyz);
    Mat rgb[3] = {r,g,b};
    split(color,rgb);
    vxl << points.cols << endl;
    vxl << "5." << endl;
    for(int i=0;i<points.cols;i++) {
      vxl << x.at<float>(0,i) << " " << y.at<float>(0,i) << " " << z.at<float>(0,i) << " " << (float)(r.at<uchar>(0,i)) << " " << (float)(g.at<uchar>(0,i)) << " " << (float)(b.at<uchar>(0,i)) << endl;
      obj << "v " << x.at<float>(0,i) << " " << y.at<float>(0,i) << " " << z.at<float>(0,i) << " " << (float)(r.at<uchar>(0,i)/255.) << " " << (float)(g.at<uchar>(0,i)/255.) << " " << (float)(b.at<uchar>(0,i)/255.) << endl;
    }
    vxl.close();
    obj.close();
    /*
      Scalar m=mean(abs(phase));
      double min,max;
      minMaxLoc(phase,&min,&max);
      cerr << "mean " << m[0] << " min " << min << " max " << max << endl;
      cerr << " create3d height " << height << " width " << width << endl;
      for(int y=0;y<height;y+=rd) {
      float planephase = 0.5f-(y-height/2.f)/zskew;
      for(int x=0;x<width;x+=rd) {

      if(mask.at<uchar>(y,x) == 0 && inpMask.at<uchar>(y,x)==0) {
      float z = (phase.at<float>(y,x)-planephase)*zscale;
      obj << "v " << x << " " << y << " " << z << endl;
      }
      }
      }
      obj.close();
      cerr << "create3d end" << endl;
    */
  }

  ~Phase() {
    ;
  }
  Phase(const Mat& i1,const Mat& i2,const Mat& i3,const Mat& msk,float skew,float scale,float thr,float contr,int skip) {
    zskew = skew;
    zscale = scale;
    noiseThr = thr;
    contrast_ = contr;
    rd = skip;
    Mat gray;
    im1 = i1.clone();
    im2 = i2.clone();
    im3 = i3.clone();
    /*
      cvtColor(i1,im1,CV_BGR2GRAY,1);
      cvtColor(i2,im2,CV_BGR2GRAY,1);
      cvtColor(i3,im3,CV_BGR2GRAY,1);

      circle(im1,Point(10,20) ,10,CV_RGB(255,255,255));
      circle(im2,Point(40,40) ,10,CV_RGB(255,255,255));
      circle(im3,Point(80,80) ,10,CV_RGB(255,255,255));
      namedWindow("X",1);
      Mat tx = im1;
      while(1) {
      imshow("X",tx);
      int k=waitKey(5);
      if(k== 'q')
      break;
      if(k=='1') 
      tx = im1;
      if(k=='2')
      tx = im2;
      if(k=='3')
      tx = im3;

      }
    */
    //cerr << im3.channels() << endl;

    //loadImage(i1,i2,i3);
    height = i1.rows;
    width = i1.cols;
    mask = Mat::zeros(im3.size(),CV_8UC1);
    inpMask = msk;
    process = Mat::zeros(im3.size(),CV_8UC1);
    phase = Mat::zeros(im3.size(),CV_32F);
  }
};

class PointCloudRenderer
{
public:
  PointCloudRenderer(const Mat& points, const Mat& img,Point3d centr,double scale);
  void reInit(const Mat& points,const Mat& img);
  void onMouseEvent(int event, int x, int y, int flags);
  void draw();
  void update(int key, double aspect);
  void setCamera(float l,float r,float t,float b,float z1,float z2,double x,double y, double z);
  int fov_;

private:
  float scale_;
  int mouse_ldx_;
  int mouse_ldy_;
		
  int mouse_rdx_;
  int mouse_rdy_;

  double phi_,theta_,r_;
  float left_;
  float right_;
  float top_;
  float bottom_;
  float znear_;
  float zfar_;
  double yaw_;
  double pitch_;
  double gamma_;
  Point3d pos_;

  TickMeter tm_;
  static const int step_;
  int frame_;

  GlCamera camera_;
  GlArrays pointCloud_;
  string fps_;
  Point3d eye_,center_,up_;
	
};

bool stop = false;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
  //cerr << "mouse callback " << x << " " << y << endl;
  if (stop)
    return;

  PointCloudRenderer* renderer = static_cast<PointCloudRenderer*>(userdata);
  renderer->onMouseEvent(event, x, y, flags);

}

void openGlDrawCallback(void* userdata)
{
  if (stop)
    return;

  PointCloudRenderer* renderer = static_cast<PointCloudRenderer*>(userdata);
  renderer->draw();
}
void PointCloudRenderer::setCamera(float l,float r,float t,float b,float z1,float z2,double x, double y, double z) {
  left_ = l;
  right_ = r;
  top_=t;
  bottom_ = b;
  znear_ = z1;
  zfar_ = z2;
  center_ = Point3d(0,0,0);
}

const int PointCloudRenderer::step_ = 20;

PointCloudRenderer::PointCloudRenderer(const Mat& points, const Mat& img, Point3d eyep, double scale)
{

  mouse_ldx_ = 0;
  mouse_ldy_ = 0;
  mouse_rdx_ = 0;
  mouse_rdy_ = 0;
  theta_ = 270;
  phi_ = 180;
  r_ = 100;
  fov_ = 140;
  yaw_ = 0.0;
  pitch_ = 0.0;
  pos_.x = 0.;
  pos_.y = 0.;
  pos_.z = -1.;
  frame_ = 0;

#ifdef LOOKAT
  eye_.x = 0.;
  eye_.y = 0.;
  eye_.y = 0.;
  eye_.z = 0.;
  eye_=eyep;
  up_.x = 0.;
  up_.y = 1.;
  up_.z = 0;
  gamma_ = 0;
  center_ = Point3d(0,-1,0);
	
  camera_.lookAt(eye_,center_,up_);
#endif
  scale_ = scale;
  camera_.setScale(Point3d(scale, scale, scale));

  pointCloud_.setVertexArray(points);
  pointCloud_.setColorArray(img, false);
  //pointCloud_.resetColorArray();
  tm_.start();
}

inline int clamp(int val, int minVal, int maxVal)
{
  return max(min(val, maxVal), minVal);
}

	
void PointCloudRenderer::onMouseEvent(int evt, int x, int y, int /*flags*/)
{
  static int Loldx = x;
  static int Loldy = y;
  static int Roldx = x;
  static int Roldy = y;
  static bool moving = false;

  //cerr << "mouse " << x << " " << y << " " << evt << " " << moving << endl;
  //theta_ = 360./500.*y;
  //phi_ = 360./600.*x;
	
  if (moving)
    {
      mouse_ldx_ = Loldx - x;
      mouse_ldy_ = Loldy - y;
      mouse_rdx_ = Roldx - x;
      mouse_rdy_ = Roldy - y;
    }
  else
    {
      mouse_ldx_ = 0;
      mouse_ldy_ = 0;
      mouse_rdx_ = 0;
      mouse_rdy_ = 0;
		
    }

  if (evt == EVENT_LBUTTONDOWN)
    {
		
      Loldx = x;
      Loldy = y;
      moving = true;
		
		
    }
  else if (evt == EVENT_LBUTTONUP)
    {
      moving = false;
    }
  if(evt ==EVENT_RBUTTONDOWN) {
    Roldx = x;
    Roldx = y;
    moving = true;
  } else if(evt == EVENT_RBUTTONUP)
    moving = false;

	
  const int mouseClamp = 300;
  mouse_ldx_ = clamp(mouse_ldx_, -mouseClamp, mouseClamp);
  mouse_ldy_ = clamp(mouse_ldy_, -mouseClamp, mouseClamp);
  mouse_rdx_ = clamp(mouse_rdx_, -mouseClamp, mouseClamp);
  mouse_rdy_ = clamp(mouse_rdy_, -mouseClamp, mouseClamp);
	
  //cerr << " mouseEvent " << mouse_ldx_ << " " << mouse_ldy_ <<  endl;
	
}

Point3d rotate(Point3d v, double yaw, double pitch)
{
  Point3d t1;
  t1.x = v.x * cos(-yaw / 180.0 * CV_PI) - v.z * sin(-yaw / 180.0 * CV_PI);
  t1.y = v.y;
  t1.z = v.x * sin(-yaw / 180.0 * CV_PI) + v.z * cos(-yaw / 180.0 * CV_PI);

  Point3d t2;
  t2.x = t1.x;
  t2.y = t1.y * cos(pitch / 180.0 * CV_PI) - t1.z * sin(pitch / 180.0 * CV_PI);
  t2.z = t1.y * sin(pitch / 180.0 * CV_PI) + t1.z * cos(pitch / 180.0 * CV_PI);

  return t2;
}

void PointCloudRenderer::update(int key, double aspect)
{
  const Point3d dirVec(0.0, 0.0, -1.0);
  const Point3d upVec(0.0, 1.0, 0.0);
  const Point3d leftVec(-1.0, 0.0, 0.0);

  const double posStep = 0.5;

  const double mouseStep = 0.008;
#ifndef LOOKAT
  camera_.setPerspectiveProjection(30.0 + fov_ / 100.0 * 40.0, aspect, 0.1, 10000.0);
	
    
  yaw_ += mouse_ldx_ * mouseStep;
  pitch_ += mouse_ldy_ * mouseStep;

  if (key == 'w')
    pos_ += posStep * rotate(dirVec, yaw_, pitch_);
  else if (key == 's')
    pos_ -= posStep * rotate(dirVec, yaw_, pitch_);
  else if (key == 'a')
    pos_ += posStep * rotate(leftVec, yaw_, pitch_);
  else if (key == 'd')
    pos_ -= posStep * rotate(leftVec, yaw_, pitch_);
  else if (key == 'q')
    pos_ += posStep * rotate(upVec, yaw_, pitch_);
  else if (key == 'e')
    pos_ -= posStep * rotate(upVec, yaw_, pitch_);
  else if(key=='+')
    scale_ += 0.01;
  else if(key=='-')
    scale_ -= 0.01;
  camera_.setCameraPos(pos_, yaw_, pitch_, 0.0);
  camera_.setScale(Point3d(scale_,scale_,scale_));

#else
  //camera_.setOrthoProjection(-10,10,-10,10,30.0,100000.);
  camera_.setPerspectiveProjection(30.0 + fov_ / 100.0 * 40.0, aspect, 0.1, 10000.0);

  if(key=='+')
    scale_ += 0.04;
  else if(key=='-')
    scale_ -= 0.04;
  else if(key == 'y')
    phi_ += posStep;
  else if(key=='t')
    phi_ -= posStep;
  else if(key == 'z')
    theta_ += posStep;
  else if(key == 'x')
    theta_  -= posStep;
  else if(key=='r') {
    eye_.x = 0.;
    eye_.y = 0;
    eye_.z =-1.;
    phi_ = 180;
    theta_ = 270;
    gamma_ = 0.;
    center_ = Point3d(0,-1,0);
    r_ = 50;
  }
  /*
  // Rotation around the X axis
  xy = Math.cos(RadianX)*y - Math.sin(RadianX)*z
  xz = Math.sin(RadianX)*y + Math.cos(RadianX)*z

  // Rotation around the Y axis
  yz = Math.cos(RadianY)*xz - Math.sin(RadianY)*x
  yx = Math.sin(RadianY)*xz + Math.cos(RadianY)*x

  // Rotation around the Z axis
  zx = Math.cos(RadianZ)*yx - Math.sin(RadianZ)*xy
  zy = Math.sin(RadianZ)*yx + Math.cos(RadianZ)*xy
  */
  phi_ += mouse_ldx_* mouseStep;
  theta_ += mouse_ldy_ * mouseStep;
  //gamma_ += mouse_rdx_ *mouseStep;
  //r_ += mouse_rdy_*mouseStep;
  //scale_ += mouse_rdx_ * mouseStep;
  camera_.setScale(Point3d(scale_,scale_,scale_));
  if(phi_ > 360.)
    phi_ = fmod(phi_,360.);
  if(theta_ > 360.)
    theta_ = fmod(theta_,360.);
  float eyeX = r_ * sin(theta_*0.0174532) * sin(phi_*0.0174532);
  float eyeY = r_ * cos(theta_*0.0174532);
  float eyeZ = r_ * sin(theta_*0.0174532) * cos(phi_*0.0174532);
  // Reduce theta slightly to obtain another point on the same longitude line on the sphere.
  float dt=1.0;
  float eyeXtemp = r_ * sin(theta_*0.0174532-dt) * sin(phi_*0.0174532);
  float eyeYtemp = r_ * cos(theta_*0.0174532-dt);
  float eyeZtemp = r_ * sin(theta_*0.0174532-dt) * cos(phi_*0.0174532); 

  float upX=eyeXtemp-eyeX;
  float upY=eyeYtemp-eyeY;
  float upZ=eyeZtemp-eyeZ;

  /*
    float x = cos(phi_*0.0174532)*r_;
    float y = sin(phi_*0.0174532)*r_;
    float z = -100.;
    float xy = cos(phi_*0.0174532)*y - sin(phi_*0.0174532)*z;
    float xz = sin(phi_*0.0174532)*y + cos(phi_*0.0174532)*z;

    // Rotation around the Y axis
    float yz = cos(theta_*0.0174532)*xz - sin((theta_*0.0174532))*x;
    float yx = sin(theta_*0.0174532)*xz + cos(theta_*0.0174532)*x;

    // Rotation around the Z axis
    //float yz = cos(theta_*0.0174532)*xz - sin((theta_*0.0174532))*x;
    float zx =cos(gamma_*0.0174532)*yx-sin(gamma_*0.0174532)*xy;
    float zy = sin(gamma_*0.0174532)*yx+cos(gamma_*0.0174532)*xy;
    float eyeX = zx; //r_ * sin(theta_*0.0174532) * sin(phi_*0.0174532);
    float eyeY = zy; //r_ * cos(theta_*0.0174532);
    float eyeZ = yz; //r_ * sin(theta_*0.0174532) * cos(phi_*0.0174532);

    float dt=1.0;
    float eyeXtemp =cos(gamma_*0.0174532-dt)*yx - sin(gamma_*0.0174532-dt)*x;//r_ * sin(theta_* 0.0174532-dt) * sin(phi_*0.0174532);
    float eyeYtemp = sin(gamma_*0.0174532-dt)*yx+cos(gamma_*0.0174532-dt)*xy;//r_ * cos(theta_*0.0174532-dt);
    float eyeZtemp = cos(gamma_*0.0174532-dt)*zx-sin(gamma_*0.0174532-dt)*x;
  */
  // Connect these two points to obtain the camera's up vector.

  Point3d up(upX,upY,upZ);

  //	pos_.z = pos_.x = pos_.y = 0.;
  //camera_.setCameraPos(pos_, yaw_, pitch_, 0.0);
  camera_.setScale(Point3d(scale_,scale_,scale_));
  Point3d eye(eyeX,eyeY,eyeZ);
  camera_.lookAt(eye,center_,up);
#endif
  tm_.stop();

  if (frame_++ >= step_)
    {
      ostringstream ostr;
      ostr << "FPS: " << step_ / tm_.getTimeSec();
      fps_ = ostr.str();

      frame_ = 0;
      tm_.reset();
    }

  tm_.start();
  //cerr << pos_ << " " << pitch_ << endl;
}

void PointCloudRenderer::draw()
{
  camera_.setupProjectionMatrix();
  camera_.setupModelViewMatrix();

  render(pointCloud_);

  render(fps_, GlFont::get("Courier New", 16), Scalar::all(255), Point2d(3.0, 0.0));
}
void PointCloudRenderer::reInit(const Mat& newpoints, const Mat& img) {
  pointCloud_.setVertexArray(newpoints);
  pointCloud_.setColorArray(img,false);
  //  camera_.setScale(Point3d(scale, scale, scale));
  PointCloudRenderer::draw();
}

void valChanged(int p,void* t) {
  int type = *(int*)t;
  if(type == 1)
    noiseChanged = true;
  changed = true;
  //cerr << type << endl;
}
void gammaChanged(int p, void* t) {
  bool *chg = (bool*)t;
  *chg = true;
}
  
  
void gammaCorr(Mat& i1, Mat& i2, Mat& i3,int gamma) {

  Mat ig1(i1.size(),CV_32F);
  Mat ig2(i2.size(),CV_32F);
  Mat ig3(i3.size(),CV_32F);
  Mat io1(i1.size(),CV_32F);
  Mat io2(i2.size(),CV_32F);
  Mat io3(i3.size(),CV_32F);
  i1.convertTo(ig1,CV_32F);
  i2.convertTo(ig2,CV_32F);
  i3.convertTo(ig3,CV_32F);
  ig1 /= 255.;
  ig2 /= 255.;
  ig3 /= 255.;
  double fgamma = gamma / 64.;
  pow(ig1,1./fgamma,io1);
  pow(ig2,1./fgamma,io2);
  pow(ig3,1./fgamma,io3);
  io1 *= 255;
  io2 *= 255;
  io3 *= 255;
  io1.convertTo(i1,i1.type());
  io2.convertTo(i2,i2.type());
  io3.convertTo(i3,i3.type());
}


void createPhaseImg(Mat& i1,Mat& i2, Mat& i3,int frq,float gamma, bool dir) {
  
  if(dir) {
    for(int i=0;i<i1.rows;i++) 
      for(int j=0;j<i1.cols;j++) {
	i3.at<uchar>(i,j) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols+(2*CV_PI/3))));
	i2.at<uchar>(i,j) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols)));
	i1.at<uchar>(i,j) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols-(2*CV_PI/3))));
      }
  } else {
    for(int j=0;j<i1.rows;j++) 
      for(int i=0;i<i1.cols;i++) {
	i3.at<uchar>(j,i) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols+(2*CV_PI/3))));
	i2.at<uchar>(j,i) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols)));
	i1.at<uchar>(j,i) = (uchar)(gamma*128.*(1.+cos(CV_PI*j*frq/i1.cols-(2*CV_PI/3))));
      }
  }
}
void createRamp(Mat& i1,int gamma) {

  for(int j=0;j<i1.rows;j++) 
    for(int i=0;i<i1.cols;i++) 
      i1.at<uchar>(j,i)=(uchar)(i*255./i1.cols);
  gammaCorr(i1,i1,i1,gamma);
}

float setVertex(const Mat& dist, Mat& points,const Mat& img, Mat& color) {

  if(points.type() != CV_32FC3) {
    points.release();
    points = Mat::zeros(dist.size(),CV_32FC3);
  }
  int nz = countNonZero(dist);
  Mat x(1,nz,CV_32FC1);
  Mat y(1,nz,CV_32FC1);
  Mat z(1,nz,CV_32FC1);
  Mat c(1,nz,CV_8UC3);
  Mat r(1,nz,CV_8UC1);
  Mat g(1,nz,CV_8UC1);
  Mat b(1,nz,CV_8UC1);
  Mat R(img.size(),CV_8UC1);
  Mat G(img.size(),CV_8UC1);
  Mat B(img.size(),CV_8UC1);
  Mat tt[3] = {B,G,R};
  split(img,tt);
  int p=0;
  for(int i=0;i<dist.rows;i++)
    for(int j=0;j<dist.cols;j++) {
      if(dist.at<float>(i,j) != 0) {
	x.at<float>(0,p)=j;
	y.at<float>(0,p)=i;
	z.at<float>(0,p) = dist.at<float>(i,j);
	r.at<uchar>(0,p) = R.at<uchar>(i,j);
	g.at<uchar>(0,p) = G.at<uchar>(i,j);
	b.at<uchar>(0,p++) = B.at<uchar>(i,j);
      }
    }
  float maxdist = 0.;
  float vxl_x=0,vxl_y=0,vxl_z =0;
  for(int i=0;i<p;i++) {
    if(x.at<float>(0,i)*x.at<float>(0,i)+y.at<float>(0,i)*y.at<float>(0,i)+z.at<float>(0,i)*z.at<float>(0,i)>maxdist*maxdist)
      maxdist = sqrt(x.at<float>(0,i)*x.at<float>(0,i)+y.at<float>(0,i)*y.at<float>(0,i)+z.at<float>(0,i)*z.at<float>(0,i));
  }
  Scalar sumx = sum(x);
  Scalar sumy = sum(y);
  Scalar sumz = sum(z);
  float dx = sumx[0]/p;
  float dy = sumy[0]/p;
  float dz = sumz[0]/p;
  x -= dx;
  y -= dy;
  z -= dz;
  Mat rgb[3] = {r,g,b};
  merge(rgb,3,c);
  Mat xyz[3] = {x,y,z};
  merge(xyz,3,points);
  color = c.clone();
  return maxdist;

}
void histog(const Mat& in) {
  int H[256];
  for(int i=0;i<256;i++)
    H[i] = 0;
  for(int i=0;i<in.rows;i++)
    for(int j=0;j<in.cols;j++) {
      uchar p=in.at<uchar>(i,j);
      H[p] += 1;
    }
  float max = 0;
  float FH[256];
  for(int i=0;i<256;i++) {
    FH[i] = (float)H[i] / (in.cols*in.rows);
    if(FH[i] > max)
      max = FH[i];
  }
  //cerr << max << endl;
  if(max == 0)
    return;
  int k=1000*max;
  Mat img(Size(256,k),CV_8UC1);
  img = Scalar::all(255);
  for(int i=0;i<256;i++)
    for(int j=0;j<k;j++) {
      Point P(j,i);
      circle(img,P,1,CV_RGB(0,0,0));
    }
  imshow("hist",img);
}




int main(int argc,char* argv[]) {

  float skew = 127.f;
  float scale = 100.f;
  float thr = 20.f;
  String i1,i2,i3;
  int skip = 2;
  int ch;
  bool files = false;
  float iScale = 1.;
  bool autom = false;
  bool vertical = false;
  bool corrCam = false;
  char *calibFile = new char [127];
  
  float gamma = 1.0;;
  double camw = 1280.,camh = 720.; 
  int frq = 30;
  Mat camMat;
  Mat distort;
  bool useMask = false;
  while((ch = getopt(argc,argv,"6as:w:t:d:i:fvg:F:cm:")) != EOF) {
    switch(ch) {
    case '6': camw = 640.; camh = 480.;
      break;
    case 'g': gamma = atof(optarg);
      if(gamma > 1.0) {
	cerr << "invalid gamma (>1)" << endl;
	gamma = 1.0;
      }
      break;
    case 'm': useMask = true;
      break;
    case 'F': frq = atoi(optarg);
      break;
    case 'v': vertical = true;
      break;
    case 'f': files = true;
      break;
    case 'i': iScale = atof(optarg);
      break;
    case 's': scale = atof(optarg);
      break;
    case 'w': skew = atof(optarg);
      break;
    case 't': thr = atof(optarg);
      break;
    case 'd': skip = atoi(optarg);
      break;
    case 'a': autom = true;
      break;
    case 'c': corrCam = true;
      strncpy(calibFile,optarg,126);
      break;
    default: cerr << "unknown parm" << endl;
      return -1;
      break;
    }
  }
  if(corrCam) {
      FileStorage fs(calibFile,FileStorage::READ);
      fs["camera_matrix"] >> camMat;
      fs["distortion_coefficients"] >> distort;
  }
  Projector pj;
  int w,h;
  bool projOK = false;
  w = 800;
  h = 600;
  if(!files) {
    if(!pj.switchCommand()) 
      cerr << "switchCommand returned false" << endl;
    if(pj.openProjector()) {
      cerr << "ok" << endl;
      pj.getDims(&w,&h);
      cerr << "resolution " << w << " " << h << endl;
      projOK = true;
    } else {

      files = true;
      cerr << "can't open projector, switching to file read mode" << endl;
    }
  }
  bool gCh = false;
  int iGamma = 155;
  namedWindow("Z",1);
  createTrackbar("gamma","Z",&iGamma,255,gammaChanged,(void*)&gCh);
  gamma = iGamma/255.;
  //waitKey(-1);
  if(!files) {
    VideoCapture cap(0); // open camera
    if(!cap.isOpened())
      return -1;
    if(cap.set(CV_CAP_PROP_FRAME_WIDTH,camw))
      cerr << "success width" << endl;
    if(cap.set(CV_CAP_PROP_FRAME_HEIGHT,camh))
      cerr << "success height" << endl;
    //if(cap.set(CV_CAP_PROP_FPS,30))
    //	cerr << "success fps" << endl;
    Mat blank = Mat::zeros(h,w,CV_8UC1);
    blank = Scalar::all((uchar)(gamma*255));
    vector<uchar>buf;
    imencode(".jpg",blank,buf);
    pj.setImage(buf,w,h,buf.size());
    Mat frame;
    while(1) {
      if(gCh) {
	gamma = iGamma/255.;
	blank = Scalar::all((uchar)(gamma*255));
	imencode(".jpg",blank,buf);
	pj.setImage(buf,w,h,buf.size());
	gCh = false;
      }
      cap >> frame;
      if(corrCam) {
	Mat temp = frame.clone();
	undistort(temp,frame,camMat,distort);
      }
      imshow("Z",frame);
      int ch = waitKey(5);
      if(ch == 'q')
	break;
    }

   
   
    Mat p1,p2,p3;
    p1 = Mat::zeros(h,w,CV_8UC1);
    p2 = Mat::zeros(h,w,CV_8UC1);
    p3 = Mat::zeros(h,w,CV_8UC1);
    
    createPhaseImg(p1,p2,p3,frq,gamma,vertical);
    //Mat ip1=imread((vertical ? "i1v.png" :"i1.png"),0);
    //Mat ip2=imread((vertical ? "i2v.png" : "i2.png"),0);
    //Mat ip3=imread((vertical ? "i3v.png" : "i3.png"),0);
    
    vector<uchar>buf2;
    vector<uchar>buf3;
    vector<uchar>buf4;
    //resize(ip1,p1,Size(p1.size()),0,0,INTER_CUBIC);
    //resize(ip2,p2,Size(p1.size()),0,0,INTER_CUBIC);
    //resize(ip3,p3,Size(p1.size()),0,0,INTER_CUBIC);
    imencode(".jpg",p1,buf);
    pj.setImage(buf,w,h,buf.size());
		
    namedWindow("P",1);
		
    imshow("P",p1);
    while(1) {
			
      //  createRamp(p1,gamma);
	
      //waitKey(10);
      cap >> frame;
      if(corrCam) {
	Mat temp = frame.clone();
	undistort(temp,frame,camMat,distort);
      }
      int k=waitKey(10);
      if(k=='q')
	break;

      //cap >> frame;
      imshow("Z",frame);
      waitKey(5);
	
    }
    cerr << "\b" << endl;
    Mat pF1,pF2,pF3,pF4;
	
    for(;;) {
      cap >> pF1;
      if(corrCam) {
	Mat temp = pF1.clone();
	undistort(temp,pF1,camMat,distort);
      }
      imshow("Z",pF1);


      int ch = waitKey(5);
      if(ch == 'q')
	break;


    }
    imwrite("px1.png",pF1);
    imencode(".jpg",p2,buf);
    pj.setImage(buf,w,h,buf.size());
	
    imshow("P",p2);  
    for(;;) {
      cap >> pF2;
      if(corrCam) {
	Mat temp = pF2.clone();
	undistort(temp,pF2,camMat,distort);
      }
      imshow("Z",pF2);


      int ch = waitKey(5);
      if(ch == 'q')
	break;


    }
    imwrite("px2.png",pF2);
    imencode(".jpg",p3,buf);
    pj.setImage(buf,w,h,buf.size());
		
    imshow("P",p3);
    for(;;) {
      cap >> pF3;
      if(corrCam) {
	Mat temp = pF3.clone();
	undistort(temp,pF3,camMat,distort);
      }

      imshow("Z",pF3);  
      int ch = waitKey(5);
      if(ch == 'q')
	break;


    }
    imwrite("px3.png",pF3);
    blank = Scalar::all(255.*gamma);
    imencode(".jpg",blank,buf);
    pj.setImage(buf,w,h,buf.size());
	
    imshow("P",blank);
    for(;;) {
      cap >> pF4;
      if(corrCam) {
	Mat temp = pF4.clone();
	undistort(temp,pF4,camMat,distort);
      }
      imshow("Z",pF4);


      int ch = waitKey(5);
      if(ch == 'q')
	break;

    }
    imwrite("px4.png",pF4);
    Mat tex;
    resize(pF4,tex,Size(800,600),0,0);
    imencode(".jpg",tex,buf);
    pj.setImage(buf,w,h,buf.size());

    pj.closeProjector();
  }
	
  Mat F1 = (files ? imread(argv[optind],0) : imread("px1.png",0));
  Mat F2 = (files ? imread(argv[optind+1],0) : imread("px2.png",0));
  Mat F3 = (files ? imread(argv[optind+2],0) : imread("px3.png",0));
  Mat F4 = (files ? imread(argv[optind+3],1) : imread("px4.png",1));

  if(!F4.empty())
    ; //F4 = imread("px4.png",1);
  else {
    F4 = Mat::zeros(F1.size(),CV_8UC3);
    F4 = Scalar::all(255);
  }
  if(F1.empty() || F2.empty() || F3.empty()) {	
    cerr << "error in imageread" << endl;
    return -1;
  }
  Mat gray,onegray;
  cvtColor(F4,gray,CV_BGR2GRAY);
  gray.convertTo(onegray,CV_8UC1);
  Mat maskSame = Mat::zeros(F1.size(),CV_8UC1);
  Mat maskLow = Mat::zeros(F1.size(),CV_8UC1);
  maskSame = abs(F1-F2) < 15;
  erode(maskSame,maskSame,Mat(),Point(-1,-1),3);
	
  maskLow = onegray < 20.;
  maskSame |= maskLow;
  if(useMask) {
    imshow("Z",maskSame*255);
    waitKey(-1);
  } else
    maskSame = 0;
  //imwrite("mask.png",maskSame);
  //system("3p.exe");
  //return 0;
#if 1

  namedWindow("OpenGL",WINDOW_OPENGL);
  resizeWindow("OpenGL",700,900);
  moveWindow("OpenGL",2200,30);
  int sk=12,sc=12,glscale = 50,dispScale = 32,nthr = 5,contr = 128;
  /*  
      i1 = argv[optind];
      i2 = argv[optind+1];
      i3 = argv[optind+2];
  */
  Scalar m;
  m=mean(F1);
  //cerr << m[0] << endl;
  m=mean(F2);
  //cerr << m[0] << endl;
  m=mean(F3);
  //cerr << m[0] << endl;
  //Mat msk = Mat::zeros(F1.size(),CV_8UC1);
  //msk = F1 < F2;
  //imshow("Z",msk);
  //waitKey(-1);
  //cerr << "mask " << countNonZero(msk) << endl;
  Phase ph(F1,F2,F3,maskSame,(sc-128)/4.,(sk-128)/4.,nthr/512.,(contr-128),4);
  ph.phaseWrap();
  ph.phaseUnwrap();
  Mat dist;
  ph.getDist(dist);
  Mat points;
  Mat colorIm;

  //setVertex(dist,points,F4,colorIm);
  float mxd = setVertex(dist,points,F4,colorIm);
  //Scalar centr = mean(points);
  Point3d eye(0,0,mxd*2.7);
  PointCloudRenderer renderer(points,colorIm,eye,0.33);

  //renderer.setCamera(minx,maxx,miny,maxy,minz,maxz,xmean,ymean,zmean);
  int ctype = 0,ntype=1;
  createTrackbar("skew","OpenGL",&sk,255,valChanged,(void*)&ctype);
  createTrackbar("scale","OpenGL",&sc,255,valChanged,(void*)&ctype);
  createTrackbar("contrast","OpenGL",&contr,255,valChanged,(void*)&ntype);
  //createTrackbar("GLscale","OpenGL",&glscale,255,valChanged);
  //createTrackbar("Dscale","OpenGL",&dispScale,255,valChanged);
  createTrackbar("Noise","OpenGL",&nthr,255,valChanged,(void*)&ntype);
  createTrackbar("Fov", "OpenGL", &renderer.fov_, 140);
  setMouseCallback("OpenGL", mouseCallback, &renderer);
  setOpenGlDrawCallback("OpenGL", openGlDrawCallback, &renderer);
  updateWindow("OpenGL");
  while(1) {
    int key =waitKey(5);
    if(key==27) 
      break;
    if (key >= 0)
      key = key & 0xff;

    double aspect = getWindowProperty("OpenGL", WND_PROP_ASPECT_RATIO);

    key = tolower(key);

    renderer.update(key, aspect);
    double min,max;
    if(changed) {
      changed = false;
      ph.setScale((float)(sk*4.),(float)(sc/4.),(float)(nthr)/512.,(float)(contr-128));
      if(noiseChanged) {
	ph.phaseWrap();	
	ph.phaseUnwrap();
	noiseChanged = false;
      }
      ph.getDist(dist);
      minMaxLoc(dist,&min,&max);
      //cerr << "min " << min << " max " << max << endl;
      if(min != 0 && max != 0) {
	setVertex(dist,points,F4,colorIm);
	//renderer.setCamera(minx,maxx,miny,maxy,minz,maxz);
	renderer.reInit(points,colorIm);
      }


    }
    //cerr << (int)key << endl;
    updateWindow("OpenGL");

  }
  ph.getDist(dist);
  setVertex(dist,points,F4,colorIm);
  //renderer.setCamera(minx,maxx,miny,maxy,minz,maxz);
  //setVertex(dist,points,F4,colorIm);
  ph.create3d("test.obj",points,colorIm);
  return 0;
#endif
}  






