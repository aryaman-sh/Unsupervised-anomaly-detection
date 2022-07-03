# Unsupervised anomaly detection
## Summer Research 2021-2022 repository
### Aryaman Sharma
<br>

The goal of this project was to work towards finding a unsupervised anomaly detection technique that

- Required Minimal labelled training data
- Produces visually interpretable results

### Literature 
- f-AnoGAN: Fast unsupervised anomaly detection with generative
   adversarial networks
   https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640
<br>
Follows up from the original AnoGAN paper.
   - Unsupervised learning on normal data to learn a latent representation.
   - Leaves the distinction and the evaluation of the capacity of detected anomalies.
   - Uses WGAN
   - Trains a GAN then an encoder based on the trained GAN model.
   - Encoder maps from latent space to image space.
   - Introduces encoder training options izi, ziz
   <br>

    ##### Observations

    - Implemented in f-AnoGAN
    - Not ideal for visual identification of anomalies
    - GAN training problems


- UNSUPERVISED REGION-BASED ANOMALY DETECTION IN BRAIN MRI
  WITH ADVERSARIAL IMAGE INPAINTING <br>
  https://arxiv.org/abs/2010.01942
    - DCNN trained to reocnstruct healthy brain regions.
    - Divide scan into regions and check reconstruciton loss. Areas with high recon loss = region with anomaly
    - Reconstruction based
    - A generator is trained to reconstruct missing regions in scans. Given a query input, performs a sliding window operation to obtain predictions for all masked regions -> Construct a heatmap indicating area of interest -> perform superpixel segmentation.


- Unsupervised Lesion Detection via Image Restoration
  with a Normative Prior <br>
  https://arxiv.org/abs/2005.00031 
  - Uses MAP estimation with a prior term learnt using VAE
  

- Unsupervised brain tumor segmentation using a symmetric-driven
  adversarial network <br>
  https://www.sciencedirect.com/science/article/abs/pii/S0925231221008262
    - The authors used c-GAN for tumor segmentation. Also suggested improvements that can be made to f-AnoGAN


- Unsupervised Anomaly Localization
  Using Variational Auto-Encoders
  https://arxiv.org/abs/1907.02796
    - Presents a comparison of AE based methods.


- MADGAN: unsupervised Medical Anomaly
  https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03936-1 <br>
  Detection GAN using multiple adjacent brain
  MRI slice reconstruction
    - This paper deals with non 2d images and trying to do anomaly detection for different stages of diseases.
    - Contains reference to implementations where only autoencoders were used
    - Uses self attention GAN


#### 
- Unsupervised anomaly detection in MR images using
  multicontrast information
- PGGAN-Based Anomaly Classification on Chest
  X-Ray Using Weighted Multi-Scale Similarity
- Improved autoencoder for unsupervised
  anomaly detection
- Fence GAN: Towards Better Anomaly Detection
- D EEP S EMI -S UPERVISED A NOMALY D ETECTION
- Deep Learning for Medical Anomaly Detection - A
  Survey
- Autoencoders for unsupervised anomaly segmentation in brain MR
  images: A comparative study
- ASC-Net: Adversarial-Based Selective
  Network for Unsupervised Anomaly
  Segmentation
- Anomaly Detection in Medical Imaging With
  Deep Perceptual Autoencoders
- Anomaly Detection in Medical Imaging - A Mini
  Review
- A Survey on GANs for Anomaly Detection


## Observations
Most current literature focuses on quantifying the extent of anomaly by calculating an 'anomaly score' this however does not always result in visually interpretable results.

Exploring reconstruction based technique especially MAP based restoration as discussed in Chen et al.

## Results
Doing MAP estimation using VAE based prior

![](images/Screenshot%20from%202022-07-01%2022-06-42.png)

From the images we can observe that although we sucessfully identify anomalous regions in the image it lacks any inductive bias towards one type of anomaly. 

