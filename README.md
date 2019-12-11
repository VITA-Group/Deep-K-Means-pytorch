## PyTorch Code for 'Deep *k*-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions'

## Introduction

PyTorch Implementation of our ICML 2018 paper ["Deep *k*-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions"](https://arxiv.org/abs/1806.09228).

[[Poster]](https://www.dropbox.com/s/covfzc7ixzt143r/ICML%2718-poster_55x33.pdf?raw=1)  
[[PPT]](https://www.dropbox.com/s/hrudc40dffg6iz1/ICML18_PPT.pdf?raw=1)

In our paper, we proposed a simple yet effective scheme for compressing convolutions though applying *k*-means clustering on the weights, compression is achieved through weight-sharing, by only recording K cluster centers and weight assignment indexes.

We then introduced a novel spectrally relaxed *k*-means regularization, which tends to make hard assignments of convolutional layer weights to K learned cluster centers during re-training. 

We additionally propose an improved set of metrics to estimate energy consumption of CNN hardware implementations, whose estimation results are verified to be consistent with previously proposed energy estimation tool extrapolated from actual hardware measurements.

We finally evaluated Deep *k*-Means across several CNN models in terms of both compression ratio and energy consumption reduction, observing promising results without incurring accuracy loss.

### PyTorch Model

- [x] Wide ResNet
- [ ] LeNet-Caffe-5

## Dependencies

Python 3.5
* [PyTorch 0.3.1](https://pytorch.org/previous-versions/)
* [libKMCUDA 6.2.1](https://github.com/src-d/kmcuda)
* sklearn
* numpy
* matplotlib


## Testing Deep k-Means

* Wide ResNet

```bash
python WideResNet_Deploy.py
```

## Filters Visualization

Sample Visualization of Wide ResNet (Conv2)

Pre-Trained Model (Before Comp.)    |  Pre-Trained Model (After Comp.)
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/Sandbox3aster/Deep-K-Means-pytorch/master/visuals/Conv2%20Pre-Trained%20Model.png)  |  ![](https://raw.githubusercontent.com/Sandbox3aster/Deep-K-Means-pytorch/master/visuals/Conv2%20Pre-Trained%20Model%20(After%20Comp.).png)
**Deep *k*-Means Re-Trained Model (Before Comp.)** | **Deep *k*-Means Re-Trained Model (After Comp.)**
![](https://raw.githubusercontent.com/Sandbox3aster/Deep-K-Means-pytorch/master/visuals/Conv2%20Deep%20k-Means%20Re-Trained%20Model%20(Before%20Comp.).png) | ![](https://raw.githubusercontent.com/Sandbox3aster/Deep-K-Means-pytorch/master/visuals/Conv2%20Deep%20k-Means%20Re-Trained%20Model%20(After%20Comp.).png)

## Citation

If you find this code useful, please cite the following paper:

    @article{deepkmeans,
        title={Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions},
        author={Junru Wu, Yue Wang, Zhenyu Wu, Zhangyang Wang, Ashok Veeraraghavan, Yingyan Lin},
        journal={ICML},
        year={2018}
    }
    
## Acknowledgment

We would like to thanks the arthor of [libKMCUDA](https://github.com/src-d/kmcuda), a CUDA based *k*-means library, without which we won't be able to do large-scale *k*-means efficiently.
