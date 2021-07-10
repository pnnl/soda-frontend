# Copyright 2021 Battelle Memorial Institute

HDF5Lib := /lustre/amat841/spack/opt/spack/linux-centos7-ivybridge/gcc-7.5.0/hdf5-1.10.7-iem767jewikudpljoztmdoxyzrtsduh7/include
CC      := g++ -g
SOURCE	:= main.cc \
		model.cc \
		util/mlir_linalg_gen.cc
FLGAS	:= -std=c++17
LD	:= -lhdf5 -L/lustre/amat841/spack/opt/spack/linux-centos7-ivybridge/gcc-7.5.0/boost-1.72.0-oaz4m24aqkonp2tab6k3yyvxstul74u7/lib -lboost_system -lboost_filesystem -lboost_program_options
TARGET	:= soda_frontend

all:
	$(CC) $(FLGAS) -I$(HDF5Lib) $(SOURCE) -o $(TARGET) $(LD)
clean:
	rm -rf $(TARGET)	
