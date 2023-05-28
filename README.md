# CV-Spring23-FDU

This is the midterm assignment of DATA130051 Computer Vision.

In the first part of the project, we use several CNNs to build classifiers for CIFAR-100, and try to use some augmentation methods like Cut-Out, Cut-Mix, and Mix-Up to optimize the model. And we finally get a top-1 accuracy of 86.79% and top-5 accuracy of 98.46% using the EfficientNetB3.

The usages of the codes like training and testing processes are shown in the corresponding folders.

## Training Steps:
Load data
Resize image from 32 by 32 into 300 by 300
Change image channel range from 0-255 into 0-1 
Augment data using Cut-Out, Cut-Mix, Mix-Up
Divide training data into training and validation set
Call model.fit(data)

## Training Steps:
Load data
Resize image from 32 by 32 into 300 by 300
Change image channel range from 0-255 into 0-1 
Use model to evaluate


## Augmentation

<img alt="Unaugmented" src="https://github.com/tianyue58/cv_assignment2_task1/assets/77108843/4d2f31c8-8e1c-44a9-b99f-817eb495d94c">

<img alt="Cut-Out" src="https://github.com/tianyue58/cv_assignment2_task1/assets/77108843/b6447391-3905-4a3b-b371-671f7b4084ef">

<img alt="Cut-Mix" src="https://github.com/tianyue58/cv_assignment2_task1/assets/77108843/71661046-544f-4d3d-810c-2de213a15159">

<img alt="Mix-Up" src="https://github.com/tianyue58/cv_assignment2_task1/assets/77108843/cc13a2ee-450b-472f-8d93-9ef3974a4906">

## Experiments

|  Dataset  |                                            Neural Network                                            | Augmentation | Accuracy | Accuracy Top 5 |
| :-------: | :--------------------------------------------------------------------------------------------------: | :----------: | :------: | :------------: |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1kyYJPV74O5c28oNOzJwi0ioXJE08pDnJ/view?usp=sharing) | Unaugmented  |  85.90%  |     98.27%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1hojlHrX3JBR-9PmjbOraSr3Y9bRUxEQ-/view?usp=sharing) |   Cut-Out    |  86.56%  |     98.49%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1XOHuIf2jssOMsdj92GVowNfIkevv3q-R/view?usp=sharing) |   Cut-Mix    |  86.79%  |     98.46%     |
| CIFAR-100 | [EfficientNetB3](https://drive.google.com/file/d/1j6GNSwVcspLvAqAB1To2UxzuE_6FUVtE/view?usp=sharing) |    Mix-Up    |  85.67%  |     98.39%     |
