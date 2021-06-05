from constellations import *
import time
import numpy as np
import cupy as cp
import numexpr as ne
#from cupyx import optimizing

fourPAM = PAM("4-PAM", 4, 1, 0)
image_ns = [1, 2, 3, 5, 10, 12]
numexpr_times = []
cupy_times = []
for image_n in image_ns:
    print("IMAGES: {}".format(image_n))
    samplesCPU = []
    for i in range(image_n) :
        samplesCPU.append(fourPAM.sampleGenerator(1000).awgn(SNR=10))
    samples = fourPAM.sampleGenerator(1000*image_n).awgn(SNR=10)


    start_time = time.time()
    for j in range(10):
        for i in range(image_n):
            samplesCPU[i].enhancedRGB((224, 224), "TIMERGB_{}.png".format(i), bounds=((-3.5, 3.5), (-3.5, 3.5)))
    numexpr_times.append(time.time() - start_time)
    #print("NUMEXPR time: {}".format(time.time() - start_time))


    start_time = time.time()
    for j in range(10):
        samples.enhancedRGBCUDABATCH((224, 224), "TIMECUDA.png", n_images=image_n, bounds=((-3.5, 3.5), (-3.5, 3.5)))
    cupy_times.append(time.time() - start_time)
    #print("CUPY time: {}".format(time.time() - start_time))

speedup = np.array(numexpr_times) / np.array(cupy_times)
print(speedup)
print(np.array(numexpr_times)/10)
print(np.array(cupy_times)/10)
plt.figure(figsize=(8, 8))
plt.plot(image_ns, speedup, '-o')
plt.xlabel('Batch size')
plt.ylabel('Speedup')
plt.title('Image generation speedup of CUDA vs NumExpr implementation for different batch sizes')
plt.show()