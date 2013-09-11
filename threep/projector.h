#include <am7xxx.h>
#include <libusb.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;

class Projector {

private:
	am7xxx_context *ctx;
	am7xxx_device *dev;
	am7xxx_device_info dev_info;
	int _width;
	int _height;
	am7xxx_image_format format;
	unsigned char* buff;
	unsigned int _size;
	bool _open;

public:
	bool switchCommand(void);
	bool openProjector(void); 
	void getDims(int* width, int *height); 

	void setImage(vector<unsigned char>&  Im, int width, int height, int size); 
	void closeProjector(); 

	Projector(void); 

	~Projector(); 
};


