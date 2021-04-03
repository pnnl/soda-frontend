SODA Front-end prototype

Dependencies: hdf5(serial), boost library

Sample run:
./soda\_frontend --arch-json examples/lenet.json --weight-h5 examples/lenet.h5 --mlir-gen translations/lenet.mlir

Common issue and solution:  
"hdf5.h: No such file or directory":  
(1) link the libhdf5\_serial.so.x.x.x as libhdf5.so under directory of /usr/lib/x86\_64-linux-gnu  
(2) same applies for libhdf5\_serial\_hl.so  

ln -s /usr/lib/x86\_64-linux-gnu/libhdf5\_serial.so.10.1.0 /usr/lib/x86\_64-linux-gnu/libhdf5.so  
ln -s /usr/lib/x86\_64-linux-gnu/libhdf5\_serial\_hl.so /usr/lib/x86\_64-linux-gnu/libhdf5\_hl.so
