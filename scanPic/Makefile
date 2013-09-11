CC = g++
CFLAGS = -Os -O2 -g
OPENCVLIBS = \
-lopencv_calib3d246.dll \
-lopencv_contrib246.dll \
-lopencv_core246.dll \
-lopencv_features2d246.dll \
-lopencv_flann246.dll \
-lopencv_gpu246.dll \
-lopencv_highgui246.dll \
-lopencv_imgproc246.dll \
-lopencv_legacy246.dll \
-lopencv_ml246.dll \
-lopencv_nonfree246.dll \
-lopencv_objdetect246.dll \
-lopencv_ocl246.dll \
-lopencv_photo246.dll \
-lopencv_stitching246.dll \
-lopencv_superres246.dll \
-lopencv_ts246 \
-lopencv_video246.dll \
-lopencv_videostab246.dll 

LDFLAGS= -Lc:/lib -lANN -Ld:/opencv-2.4.6.1/build-m64-ocl/install/lib $(OPENCVLIBS)
INCLUDES = -Ic:/include -Id:/opencv-2.4.6.1/build-m64-ocl/install/include
SOURCE = LoOP.cpp scan.cpp Serial.cpp
OBJS = LoOP.o scan.o Serial.o

all:	scanner

scanner:	$(OBJS)
	$(CC) -o scanner.exe $(CFLAGS) $(OBJS) $(LDFLAGS)


$(OBJS):	%.o:%.cpp
	$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@ 

clean:
	rm -f *.o scanner.exe


