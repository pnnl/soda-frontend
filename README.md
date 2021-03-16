Front-end prototype

Dependencies: hdf5(serial), boost library

Sample run:
./main examples/lenet.json examples/lenet.h5

Common issue and solution:  
"hdf5.h: No such file or directory":  
(1) link the libhdf5\_serial.so.x.x.x as libhdf5.so under directory of /usr/lib/x86\_64-linux-gnu  
(2) same applies for libhdf5_serial_hl.so  

ln -s /usr/lib/x86\_64-linux-gnu/libhdf5\_serial.so.10.1.0 /usr/lib/x86\_64-linux-gnu/libhdf5.so  
ln -s /usr/lib/x86\_64-linux-gnu/libhdf5\_serial\_hl.so /usr/lib/x86\_64-linux-gnu/libhdf5\_hl.so
