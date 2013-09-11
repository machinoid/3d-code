#include "projector.h"

bool Projector::openProjector(void) {
  int st = am7xxx_init(&ctx);
  if(st<0) {
	  perror("init");
    return false;
  }
  st = am7xxx_open_device(ctx, &dev,0);
  if(st<0) {
    perror("can't open projector");
    return false;
  }
  st = am7xxx_get_device_info(dev,&dev_info);
  if(st<0) {
    perror("can't get dev info");
    return false;
  }
  _open = true;
  return true;
}
void Projector::getDims(int* width, int *height) {
  if(_open == false) {
    cerr << "open projector first" << endl;
    *width = 0;
    *height = 0;
    return;
  }
  _width = dev_info.native_width;
  _height = dev_info.native_height;
  *height = _height;
  *width = _width;
}

void Projector::setImage(vector<unsigned char>&  Im, int width, int height, int size) {
  if(width == 0 || height == 0) {
    cerr << "zero image" << endl;
    return;
  }
  if(_width ==0) {
    int tmpw,tmph;
    getDims(&tmpw,&tmph);
  }
  if(width > _width || height > _height) {
    cerr << "image too large" << endl;
    return;
  }
  
  
  if(Im.size() <= size) {
    buff = &Im[0];
    //std::copy(Im.begin(),Im.end(),buff);
  } else {
    cerr << "size doesn't match" << endl;
    return;
  }
  
  int st = am7xxx_send_image(dev,AM7XXX_IMAGE_FORMAT_JPEG,width, height,buff,size);
  if(st<0) {
    cerr << "error displaying image" << endl;

    return;
  }

}

	
void out_libusb_close(libusb_device_handle *usb_device, int interf) {
	libusb_release_interface(usb_device, interf);
	libusb_close(usb_device);
	//usb_device = NULL;
}

bool Projector::switchCommand(void) {
	
#define AM7XXX_STORAGE_VID           0x1de1
#define AM7XXX_STORAGE_PID           0x1101
#define AM7XXX_STORAGE_CONFIGURATION 1
#define AM7XXX_STORAGE_INTERFACE     0
#define AM7XXX_STORAGE_OUT_EP        0x01

static unsigned char switch_command[] =
	"\x55\x53\x42\x43\x08\x70\x52\x89\x00\x00\x00\x00\x00\x00"
	"\x0c\xff\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

	int ret;
	int transferred;
	unsigned int vid,pid;
	vid = AM7XXX_STORAGE_VID;
	pid = AM7XXX_STORAGE_PID;
	
	  
	libusb_device_handle *usb_device = NULL;

	unsigned int len;

	ret = libusb_init(NULL);
	if (ret < 0)
		libusb_exit(NULL);
		return false;

	libusb_set_debug(NULL, 0);

	usb_device = libusb_open_device_with_vid_pid(NULL,
						     vid,
						     pid);
	if (usb_device == NULL) {
		cerr <<  "cannot open the device: " << endl;
		libusb_exit(NULL);
		return false;
	}

	if (libusb_kernel_driver_active(usb_device, AM7XXX_STORAGE_INTERFACE)) {
		ret = libusb_detach_kernel_driver(usb_device,
						  AM7XXX_STORAGE_INTERFACE);
		if (ret < 0)
			cerr << "Warning: cannot detach kernel driver" << endl;
	} else {
		cerr << "kernel driver not active" << endl;
	}

	ret = libusb_set_configuration(usb_device, AM7XXX_STORAGE_CONFIGURATION);
	if (ret < 0) {
		cerr << "cannot set configuration" << endl;
		out_libusb_close(usb_device, AM7XXX_STORAGE_INTERFACE);
		return false;
	}

	ret = libusb_claim_interface(usb_device, AM7XXX_STORAGE_INTERFACE);
	if (ret < 0) {
		cerr << "cannot claim interface" << endl;
		out_libusb_close(usb_device, AM7XXX_STORAGE_INTERFACE);
		return false;
	}

	len = sizeof(switch_command);
	transferred = 0;
	ret = libusb_bulk_transfer(usb_device, AM7XXX_STORAGE_OUT_EP,
				   switch_command, len, &transferred, 0);
	if (ret != 0 || (unsigned int)transferred != len) {
		fprintf(stderr, "ret: %d\ttransferred: %d (expected %u)\n",
		      ret, transferred, len);
		out_libusb_close(usb_device, AM7XXX_STORAGE_INTERFACE);
		return false;
	}

	fprintf(stderr, "OK, command sent!\n");
	libusb_exit(NULL);
	return true;
}

void Projector::closeProjector() {
  if(_open)
    am7xxx_shutdown(ctx);
  _open = false;
}

Projector::Projector(void) {
  _width = 0;
  _height = 0;
  buff = NULL;
  _open = false;
}

Projector::~Projector() {
 
  if(_open) {
    am7xxx_shutdown(ctx);
  }
}
