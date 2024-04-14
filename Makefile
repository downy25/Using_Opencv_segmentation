CX = g++
CXFLAGS = -g -Wall 

CVFLAGS = `pkg-config opencv4 --cflags --libs`

BUILDFLAGS = $(CVFLAGS)

TARGET = segmentation
OBJS = main.o
$(TARGET) :  $(OBJS)
	$(CX) $(CXFLAGS) -o $(TARGET) $(OBJS) $(BUILDFLAGS) 
main.o : main.cpp
	$(CX) $(CXFLAGS) -c main.cpp $(BUILDFLAGS)

.PHONY: all clean
all: $(TARGET)

clean:
	rm -rf $(TARGET) $(OBJS)