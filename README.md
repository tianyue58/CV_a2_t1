# CV-Spring23-FDU

This is the midterm assignment of DATA130051 Computer Vision.

In the first part of the project, we use several CNNs to build classifiers for CIFAR-100, and try to use some augmentation methods like Cut-Out, Cut-Mix, and Mix-Up to optimize the model. And we finally get a top-1 accuracy of 86.79% and top-5 accuracy of 98.46% using the EfficientNetB3.

The usages of the codes like training and testing processes are shown in the corresponding folders.

## Training Steps:
1. Load data
2. Resize image from 32 by 32 into 300 by 300
3. Change image channel range from 0-255 into 0-1 
4. Augment data using Cut-Out, Cut-Mix, Mix-Up
5. Divide training data into training and validation set
6. Call model.fit(data)

## Training Steps:
1. Load data
2. Resize image from 32 by 32 into 300 by 300
3. Change image channel range from 0-255 into 0-1 
4. Use model to evaluate


## Augmentation

<img alt="Unaugmented" src="https://github.com/tianyue58/CV_a2_t1/assets/77108843/09189826-2fbd-4b62-ab08-f84092c68c23">

<img alt="Cut-Out" src="https://github.com/tianyue58/CV_a2_t1/assets/77108843/255b6c57-9ccf-4d38-b2b0-209de3509b90">

<img alt="Cut-Mix" src="https://github.com/tianyue58/CV_a2_t1/assets/77108843/7cf83562-e5f5-4336-ae6a-4d7e783774ce">

<img alt="Mix-Up" src="https://github.com/tianyue58/CV_a2_t1/assets/77108843/ffd419a8-5f90-490f-8aa2-080838122448">

## Experiments

|  Dataset  |                                            Neural Network                                            | Augmentation | Accuracy | Accuracy Top 5 |
| :-------: | :--------------------------------------------------------------------------------------------------: | :----------: | :------: | :------------: |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1kyYJPV74O5c28oNOzJwi0ioXJE08pDnJ/view?usp=sharing) | Unaugmented  |  85.90%  |     98.27%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1hojlHrX3JBR-9PmjbOraSr3Y9bRUxEQ-/view?usp=sharing) |   Cut-Out    |  86.56%  |     98.49%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1XOHuIf2jssOMsdj92GVowNfIkevv3q-R/view?usp=sharing) |   Cut-Mix    |  86.79%  |     98.46%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1j6GNSwVcspLvAqAB1To2UxzuE_6FUVtE/view?usp=sharing) |    Mix-Up    |  85.67%  |     98.39%     |
