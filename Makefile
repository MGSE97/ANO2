CXX=g++
CFLAGS = -g
OPENCV_PATH = `pkg-config --cflags --libs opencv`

all: ANO 

ANO: ANO.cpp
	$(CXX) $(CFLAGS) -o ANO ANO.cpp $(OPENCV_PATH)

.PHONY:	clean
clean:
	rm -f *~ *.o ANO
