# Attention Filter Gate

***Welcome to the Attention Filter Gate repository! Here, we provide an implementation of our proposed method, the Attention Filter, which is based on the Fast Fourier Transform. In the accompanying PDF document, we explain in detail the steps we have taken to tackle the problem at hand.***

### Dataset Description

The data used in this project is hosted by a competition on the Kaggle platform, namely the Left Atrial Segmentation Challenge. We have used gmedical images of the left atrium, which are in 3D and come with 30 corresponding masks.

* Data Source :</br>

    The Data we have been used is Host in Kaggle Platform competiotion , to download Run following command :
    there're few step needed to be consider before running the Script 

    1. **First**:

          - Create an account in Kaggle and Get API Token 
    2. **Second**:
    
          - Replace your Kaggle API Tokon which stored in **kaggle.json** in the right path in Script to have authrozation and following this command :
        
                    chmod a+x download.sh && ./download.sh
        
    3. the Virtualization Sample: 

         ![virtualization](Figures/image_label_overlay_animation.gif)
### Introduction

In this project, we aim to build a new mechanism, the Attention Filter Gate, which will address the weaknesses of previous approaches used to handle certain problems, such as:
- **Critical Problems we assign** 
    * Losing features during extraction when using deep segmentation models
    * Handling data with higher resolutions
    * Reducing time complexity of training
    * Reducing energy consumption during training
    * Learning from spatial domain

Through our exploration of these weaknesses, we aim to provide a better solution to these problems using our proposed Attention Filter Gate mechanism.

### Main Abstarct Thesis :
* **Abstract**
    * Background: 

        Medical imaging diagnosis can be challenging due to low-resolution images caused
        by machine artifacts and patient movement. Researchers have explored algorithms
        and mathematical insights to enhance image quality and representation. One com-
        mon task is segmentation, which requires the detection or localization of diseases
        around the tissue. New approaches and models using artificial intelligence, specif-
        ically computer vision, have been developed to improve the traditional methods
        that have not been entirely effective. Since the publication of the U-Net model
        paper, researchers have focused on building new model architectures to segment
        medical images more effectively. The Transformers model, a core technology be-
        hind many AI applications today, has been a game-changer. The Attention Gate
        was introduced in a paper and used with the U-Net model to increase performance.
        However, it did not solve certain computational cost issues and led researchers to
        investigate how to improve the Attention Gate in a different way while maintaining
        the same structure.
    * Aim : 

        The aim was to improve the existing Attention Gate used in U-Net for medical
        image segmentation. The goal was to reduce the computational cost of training
        the model, improve feature extraction, and handle the problem of matrix multi-
        plication used in CNN for feature extraction
        Method
        The Attention Filter Gate was developed to improve upon the Attention Gate.
        Instead of learning from the spatial domain, the model was converted to the fre-
        quency domain using the Fast Fourier Transformation (FFT). A weighted learnable
        matrix was used to filter features in the frequency domain, and FFT was imple-
        mented between up-sampling and down-sampling to reduce matrix multiplication.
        The method tackled computational cost, complexity algorithm, throughput, la-
        tency, FLOP, and enhanced feature extraction.
    * Results
    
        Describe the main results of after finishing some Quantitative results empty for
        now
    * Conclusion:

        This thesis investigates the Attention Filter Gate to address problems such as
        computational cost and feature extraction, providing an alternative approach to
        medical image segmentation that is both efficient and effective. The method en-
        hances feature extraction to reduce information loss between the encoder and
        decoder, and it provides a potential solution for throughput, latency, FLOP, and
        algorithm complexity issues. The Attention Filter Gate improves on the existing
        Attention Gate with intuitive tricks not addressed by previous methods.
    * Keywords:

        Medical Segmentation, Neural networks, Transformers, U-Net model, Attention
        Gate , Fast Fourier Transformation (FFT)
