# Implementation of Denoising Algorithms on CIFAR-10 Dataset

- Applied Various Unsupervised Machine Learning algorithms on ciar-10 data to denoise the images. 
- Focused on Denoising auto-encoders which does unsupervised pre-training and can learn good representations via reconstruction
-  To prevent the auto-encoders from merely copying inputs during training, denoising auto-encoders were proposed to learn representations from corrupted data

## Algorithms Used :  
1.  Simple Denoising Autoencoders (DAE)
2. Convolutional Auto-encoders with Symmetric Skip Connections
3. Feed-forward denoising convolutional neural networks (DnCNNs)

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
 - Encoder :  3x3 convolutional layers and d ownsamping done using 2x2 maxpooling layers
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
Batch Size = 20
no. of epochs = 40
Learning rate =0.001

### **RESULTS**
![simres.png](/images/simres.PNG)

### **EVALUATION**
On Test data of 10000 examples:
- Average **PSNR**:24.830 
- Average **SSIM**: 0.868

