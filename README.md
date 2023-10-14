
# One pixel attack for fooling deep neural network

Adversarial attacks are intriguing vulnerabilities in deep neural networks (DNNs) that reveal their susceptibility to seemingly minor perturbations in input data. A captivating 
example of such an attack is the "one-pixel attack," where an attacker modifies just a single pixel in an image to trick the neural network into making a wrong classification 
decision. This interview discussion will delve into the nuances of the one-pixel attack,its underlying techniques, challenges faced, and its broader implications.

The one-pixel attack aims to explore the susceptibility of DNNs to imperceptible changes in input images. The challenge is to select a single pixel in an image and determine its 
color values to achieve a targeted misclassification while ensuring the modified pixel remains visually indistinguishable from the original.

The core approach involves using an optimization algorithm, such as Differential Evolution,to iteratively adjust the color values of a chosen pixel. This optimization maximizes the 
likelihood of a desired misclassification while obeying constraints for imperceptibility.The objective function guides the optimization, combining probabilities from the network's
softmax outputs to emphasize the target misclassification.

## Dataset
 CIFAR-10 is a widely used dataset for image classification tasks. It consists of
60,000 color images divided into 10 classes, with 6,000 images per class.
Each image in CIFAR-10 has a resolution of 32x32 pixels. The dataset
includes diverse objects such as airplanes, cars, birds, cats, and more.
## Key feature
- Objective function:
   
   The Objective function is a critical component of the one-pixel attack.It quantifies the sucess of the attack by provising a measure how well the manipulated image align with the attacker goal.  
   In the content of attack, the objective function evaluates hw close the modified image is to acheiving the desired misclassification. This function guides the optimization process by providing a score that indicate the quality of perturbation applied to the chosen pixel.  
  The Objective funtion might be design to maximize the output probabilit of a specific target class while minimizing the probabilites of other class.

- Optimization Algorithm: 
  
  The optimization algorithm is responsible fo iteratively adjusting the color value of the chosen pixel to maximize the value of the objective function while adhering to constraints. Differential Evolution is a popular optimization algorithm used in the one-pixel attack. It creates candidate solutions by combining information from multiple solutions and generates trial solutions that are evaluated based on the objective function.  
  The optimization algorithm explores the space of possible color values for the chosen pixel, seeking the values that lead to the highest likelihood of achieving the desired misclassification. It iterates through multiple rounds of mutation, crossover, and evaluation to refine the pixel values.
  The optimization process needs to strike a balance between maximizing the likelihood of misclassification and staying within the imperceptibility constraints. Ensuring that the perturbation is not overly noticeable to human observers is a key aspect of crafting an effective one-pixel attack.


- Pixel Selection:
  
  Pixel selection involves identifying a single pixel within the input image to be manipulated. The choice of this pixel can significantly impact the effectiveness of the attack. There are different strategies for selecting the pixel to maximize the likelihood of a successful misclassification.  
  Some strategies might involve selecting pixels that are part of regions in the image known to be crucial for classification, while others might use gradient-based methods to identify pixels that have a higher impact on the network's output. The selected pixel's color values are then modified through the optimization process to achieve the desired misclassification while adhering to constraints.

## Roadmap
- The one-pixel attack aims to manipulate the output of the model by changing the color of a single pixel in the input image. This manipulation is done in such a way that the model's classification result is altered.

- The attacker doesn't need access to the model's internal parameters or gradients. Instead, they interact with the model purely by providing an input image and observing the model's predictions.

- The attack is an optimization problem. The attacker iteratively modifies the color of the pixel and observes how these changes impact the model's prediction. The goal is to find the optimal color for the single pixel that will make the model classify the image into a different, predefined target class.

- The attack algorithm uses a gradient-based or evolutionary approach to determine the most effective change to the pixel color. It computes how the gradient of the model's output probability with respect to the pixel's color should be adjusted to achieve the target class. This process aims to make the model as confident as possible in the desired misclassification.

- The one-pixel attack demonstrates that deep neural networks, even when highly accurate, can be surprisingly vulnerable to minor changes in the input data. It underscores the importance of developing more robust models and raises awareness about the security and reliability of deep learning systems in real-world applications.-

## Model
Dataset - **CIFAR-10**  
Accuracy - **85%**

```
----------------------------------------------------------------

"""
input   - (3, 32, 32)
block 1 - (32, 32, 32)
maxpool - (32, 16, 16)
block 2 - (64, 16, 16)
maxpool - (64, 8, 8)
block 3 - (128, 8, 8)
maxpool - (128, 4, 4)
block 4 - (128, 4, 4)
avgpool - (128, 1, 1), reshpe to (128,)
fc      - (128,) -> (10,)
"""

# block
Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
ReLU()
Conv2d(32, 32, kernel_size=3, padding=1)
BatchNorm2d(32)
ReLU()

#
MaxPool2d(kernel_size=2, stride=2)

# avgpool
AdaptiveAvgPool2d(1)

# fc
Linear(256, 10)

----------------------------------------------------------------
```
## Result 
One-pixel attacks have shown that DNNs can be easily fooled by imperceptible 
modifications. These attacks underscore the need for robust model architectures and 
adversarial training to enhance network resilience against such vulnerabilities.

| ![airplane_bird_8075](https://github.com/sachin576123/One-Pixel-Attack-for-fooling-depp-neural-network/assets/33089431/08992279-9098-438d-a38a-2376cf46d5a3) | ![bird_deer_8933](https://github.com/sachin576123/One-Pixel-Attack-for-fooling-depp-neural-network/assets/33089431/9e195a8c-d0b4-4dcd-b9b1-bfec4999c0f7) | ![cat_frog_8000](https://github.com/sachin576123/One-Pixel-Attack-for-fooling-depp-neural-network/assets/33089431/1db35ef8-8ce7-40f7-894f-a7c67408f0d5)  |        ![frog_bird_6866](https://github.com/sachin576123/One-Pixel-Attack-for-fooling-depp-neural-network/assets/33089431/0144c0bc-caa7-4a95-9f3c-cf5b4fe29ec2) |  ![horse_deer_9406](https://github.com/sachin576123/One-Pixel-Attack-for-fooling-depp-neural-network/assets/33089431/d684ce1a-685a-4d46-9988-a8086bd0ad85) |
|:------------------------------------------:|:----------------------------------:|:---------------------------------:|:-----------------------------------------:|:--------------------------------------:|  
| **bird [0.8075]**                   |               **deer [0.8933]**           |  **frog [0.8000]**                |                        **bird [0.6866]**   |       **deer [0.9406]**                |


## Tech Stack

**Language:** Python

**Framework:** PyTorch

**Libraries:** Numpy,Matplotlib


## Defense Mechanism
 
 - Adversarial Training: Train DNNs using a combination of clean and adversarial examples, including those generated by the one-pixel attack. This can help the model become more robust by learning to recognize and differentiate between genuine and perturbed inputs.

 - Regularization Techniques: Apply regularization methods such as L2 regularization, dropout, and weight decay during training to prevent the model from being overly sensitive to small input changes.
## Learning
 This project highlights the vulnerability of deep learning models to minor input perturbations. By manipulating a single pixel in an image, it demonstrates that seemingly insignificant changes can lead to incorrect classifications. This project underscores the importance of model robustness and security in real-world applications, necessitating the development of defenses against adversarial attacks. It teaches us to critically assess the limitations of the machine learning models and tend to develop more robust defense mechanism.
## Support
 For support, email sachin576123@gmail.com.
















