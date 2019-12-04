# csci5551
g++ -O quickSort_s.cpp -o quickSort_s

ssh node18 nvcc -arch=sm_60 -rdc=true /home/jason.2.zhang/csc5551/project/quickSort/quickSort_gpu.cu -o /home/jason.2.zhang/csc5551/project/quickSort/quickSort_gpu

g++ -O bucketSort_s.cpp -o bucketSort_s

ssh node18 nvcc -arch=sm_60 -rdc=true /home/jason.2.zhang/csc5551/project/bucketSort/bucketSort_gpu.cu -o /home/jason.2.zhang/csc5551/project/bucketSort/bucketSort_gpu
