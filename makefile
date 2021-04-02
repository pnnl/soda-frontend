HDF5Lib := /usr/include/hdf5/serial/
CC      := g++
SOURCE	:= main.cc \
		model.cc
FLGAS	:= -std=c++17
LD	:= -lhdf5 -lboost_system -lboost_filesystem -lboost_program_options
TARGET	:= soda_frontend

all:
	$(CC) $(FLGAS) -I$(HDF5Lib) $(SOURCE) -o $(TARGET) $(LD)
clean:
	rm -rf $(TARGET)
