HDF5Lib := /usr/include/hdf5/serial/

all:
	g++ -std=c++17 -I$(HDF5Lib) main.cc model.cc -o main -lhdf5 -lboost_system -lboost_filesystem
clean:
	rm -rf main
