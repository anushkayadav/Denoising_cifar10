# Implementation of Denoising Algorithms on CIFAR-10 Dataset

- Applied Various Unsupervised Machine Learning algorithms on ciar-10 data to denoise the images. 
- Focused on CNN based approaches which do unsupervised pre-training and can learn good representations via reconstruction
- To prevent the auto-encoders from merely copying inputs during training, denoising auto-encoders were proposed to learn representations from corrupted data

<p align="center">
 <b>Framework Used :</b>
<a href="https://pytorch.org/"><img src="https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/pytorch-logo-dark.png" width="100" /></a>
</p>

## Algorithms Used :  
There are broadly two types of deep learning algorithms which may be used for denoising :

- Discriminative Learning-Based using CNNs
- Generative Learning-Based using GANs

For the denoising problem of known noise like **Gaussian noise**, using **CNNs based approaches** it is possible to form paired training data and leverage these methods to achieve state-of-the-art performance. They could fully exploit the great capability of the network architecture to learn from data, which breaks through the limitations of prior based methods and further improves the performance whereas **GANs** are used where there are more **complex real noises** and dataset is small.[[1]](#1)

Since, in our experiment we used **simple Gaussian noise** and CIFAR-10 dataset has **considerable amount of data**(60000 examples), I preferred to use Discriminative Learning-Based such as:
1. Simple Denoising Autoencoders (DAE) [Reference Paper](https://www.researchgate.net/publication/330382260_Image_Denoising_with_Color_Scheme_by_Using_Autoencoders)[[2]](#2)
2. Convolutional Auto-encoders with Symmetric Skip Connections [Reference Paper](https://arxiv.org/pdf/1611.09119.pdf)[[3]](#3)
3. Feed-forward denoising convolutional neural networks (DnCNNs) [Reference Paper](https://arxiv.org/pdf/1608.03981.pdf)[[4]](#4)

## Dataset and Noise
The dataset used  comprises of 60000 color pictures in 10 classes with 6000 picture per class.
Dimension of each image is 32 x 32.

I have introduced External Noise i.e. **Pixel-level Gaussian noise** to all the input images which would be fed into our models.

We add Gaussian noise matrix on both training and testing with noise factor 0.1 and clip images between 0 and 1.
```
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
```
![im1.png](/images/im1.PNG) 
## Simple Denoising Autoencoder (DAE)

### **MODEL ARCHITECTURE**
 - Encoder :  3x3 convolutional layers and downsamping done using 2x2 maxpooling layers
 - Decoder : Upsampling done and layers are symmetric to the encoder. 
 
Each convolutional/deconvolutional layer is followed by a **ReLU** non-linearity layer
 
```
ConvDenoiser(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (t_conv1): ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))
  (t_conv2): ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))
  (convout): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```
**Hyperparameters** : 
Batch Size = 20 **.**
no. of epochs = 40 **.**
Learning rate =0.001 **.**

### **RESULTS**
![simres.png](/images/simres.PNG)

### **EVALUATION**
On Test data of 10000 examples:
- Average **PSNR**:24.830 
- Average **SSIM**: 0.868

## Convolutional Auto-encoders with Symmetric Skip Connections

### **MODEL ARCHITECTURE**
- Encoder :  3x3 convolutional layers and downsamping done using stride of 2 instead of pooling, which can be harmful to image restoration tasks
- Decoder : Upsampling done and layers are symmetric to the encoder. 
- The corresponding encoder and decoder layers are connected by **shortcut connections**.
- Each convolutional/deconvolutional layer is followed by a **Batch Normalization** layer and a
**ReLU non-linearity** layer
- Number of encoder/decoder layers : 15
<p align="center"><img src="/images/rednet30_arc.PNG" width="300" /></p>
 

**Hyperparameters** : 
Batch Size = 32 **.**
no. of epochs = 40 **.**
Learning rate =0.001 **.**

### **RESULTS**
![simres.png](/images/redres.PNG)

### **EVALUATION**
On Test data of 10000 examples:
- Average PSNR:28.254 
- Average SSIM: 0.938

## Feed-forward denoising convolutional neural networks (DnCNNs)
### **MODEL ARCHITECTURE**
- VGG modeified network
- Integration of Residual Learning and Batch Normalization
- Size of convolutional filters to be 3 × 3 but remove all pooling layers.
3 Types of layers used : 
1. **Conv+ReLU** : generate feature maps
2. **Conv+BN+ReLU** : batch normalization incorporated to speed up training as well as boost the denoising performance
3.  **Conv** : To reconstruct the output
<p align="center"><img src="/images/dnnarc.PNG" width="400" /></p>

**Hyperparameters** : 
Batch Size = 32  **.**
no. of epochs = 40  **.**
Learning rate =0.001  **.**

### **RESULTS**
![simres.png](/images/dncnres.PNG)

### **EVALUATION**
On Test data of 10000 examples:
- Average PSNR:28.992 
- Average SSIM: 0.947
## Conclusion
**SSIM** (Structural Similarity Image Metric), which estimates the degradation of structural similarity based on the statistical properties of local information between a reference and a distorted image.It combines three local similarity measures based on luminance, contrast, and structure.

**PSNR**, the term peak signal-to-noise ratio is an expression for the ratio between the maximum possible value (power) of a signal and the power of distorting noise that affects the quality of its representation
The main limitation of this metric is that it relies strictly on numeric comparison and does not actually take into account any level of biological factors of the human vision system such as the structural similarity index. (SSIM)
<p align="center"><img src="/images/table.PNG" width="500" /></p>


## References
- <a id="1">[[1] J. Chen, J. Chen, H. Chao and M. Yang, "Image Blind Denoising with Generative Adversarial Network Based Noise Modeling," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 3155-3164, doi: 10.1109/CVPR.2018.00333.](https://ieeexplore.ieee.org/document/8578431)</a>

- <a id="2">[[2] Ali, Irfan & Lashari, Haque & Hassan, Syed & Maitlo, Abdullah & Qureshi, Basit. (2018). Image Denoising with Color Scheme by Using Autoencoders. 18. 158-161.](https://www.researchgate.net/publication/330382260_Image_Denoising_with_Color_Scheme_by_Using_Autoencoders)</a>

- <a id="3">[[3] Mao, Xiaojiao, Chunhua Shen, and Yu-Bin Yang. "Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections." Advances in neural information processing systems. 2016.](http://papers.nips.cc/paper/6172-image-restoration-using-very-deep-convolutional-encoder-decoder-networks-with-symmetric-skip-connections)</a>

- <a id="4">[[4] Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing, 26(7), 3142-3155.](https://ieeexplore.ieee.org/abstract/document/7839189)</a>

## Other Refrences

- [Fan, L., Zhang, F., Fan, H. et al. Brief review of image denoising techniques. Vis. Comput. Ind. Biomed. Art 2, 7 (2019). https://doi.org/10.1186/s42492-019-0016-7](https://link.springer.com/article/10.1186/s42492-019-0016-7)
- [Gu, Shuhang, and Radu Timofte. "A brief review of image denoising algorithms and beyond." Inpainting and Denoising Challenges. Springer, Cham, 2019. 1-21.](https://link.springer.com/chapter/10.1007/978-3-030-25614-2_1)
- []()






