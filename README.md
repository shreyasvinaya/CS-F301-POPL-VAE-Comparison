# CS F301 POPL - Variational Autoencoders Generative Modelling using Pyro vs Pytorch

## Group Members- Group 29

| Name of member | BITS ID | Email |
|---|---|---|
| Anish Sreenivas | 2020B3A71464G | f20201464@goa.bits-pilani.ac.in |
| Shreyas V | 2020B2A71350G | f20201350@goa.bits-pilani.ac.in |
| Pranav Srikanth | 2020B5A71865G | f20201865@goa.bits-pilani.ac.in |
| Nandish Chokshi | 2020B1A72031G | f20202031@goa.bits-pilani.ac.in |


## Problem statement
In this project, we aim to implement Variational Autoencoders Generative Modelling using Pyro vs Pytorch

> Where is the POPL angle in it?

* We are specifically exploring two of the guidelines given to us: reliability and easy-of-use
- Reliability- Reduces possibility of data type mismatches because we are 
using inference algorithms.
- Ease of use- Only the model and the guide specification is needed to run 
the optimizer (the objective function does not need to be specified as in 
the PyTorch implementation).

> Was it solved before? How is your solution different?

## Executing the Code

Clone the repository
```bash
git clone https://github.com/shreyasvinaya/CS-F301-POPL-VAE-Comparison.git
```

Installing Dependencies
```bash
pip install -r requirements.txt
```

<hr>

Running the code for Pyro
```bash
python main.py --impl pyro
```

Running the code for Pytorch
```bash
python main.py --impl pytorch
```


- **Aim**: is to show improvement in reliability and ease of use for the developer to generate images.
- **Reliability**: Reduces the possibility of data type mismatches because we are using inference algorithms.
- **Ease of use**: Only the model and the guide specification are needed to run the optimizer (the objective function does not need to be specified as in the PyTorch implementation).
---
## Software architecture

> What is the software architecture of your soln? What parts have you reused and what parts have you developed on your own? Draw a figure to explain better. Is it a client-server architecture. Where is the testing component placed (local or remote)? Is there a database involved? etc.

This is a comparitive study using two python libraries:
1. pyro
2. pytorch

* The code runs completely locally with no servers (no client-server architecture is needed)

---
## POPL Aspects

> What were the POPL aspects involved in the implementation. NOT theoretical answers. Have pointers to the lines of code and explain the POPL ideas/concepts involved and why they are necessary. I expect 5 to 10 points written on POPL aspects (bullet points, one after another). More the points you have the better it is. While writing the points also write your experience of the difficulties you faced.

1. **Abstraction (Lines 16-48, 66-100, 102-144):** 
    - *Encapsulation:* The code implements classes such as `Encoder`, `Decoder`, and `VAE`, encapsulating specific functionalities related to encoding, decoding, and the Variational Autoencoder (VAE) itself.
    - *Inheritance and Polymorphism:* The inheritance hierarchy is seen in the classes `VAE`, `PyTorchVAEImpl`, and `PyroVAEImpl`, where subclasses inherit behavior from the superclass (`VAE`) and implement their specific functionalities.

2. **Data Types and Typing (Lines 8-10, 20-22, 24-27, 30-33, 36-39):** 
    - The code uses type annotations (`: tuple[Any, torch.Tensor]`, `-> None`, etc.) for function signatures, indicating the expected input and output types, enhancing code readability and enabling type checking tools.

3. **Functional and Imperative Paradigms (Lines 50-65, 102-144):** 
    - The code combines both paradigms. The imperative paradigm is employed during training and testing loops (`for` loops, calling `.backward()` and `.step()` for optimization), while the functional paradigm is utilized for defining the forward pass in neural networks and loss computations.

4. **Dynamic and Static Dispatch (Lines 70-73, 125-128):** 
    - Dynamic dispatch is used for selecting the forward pass of the appropriate VAE implementation (`PyTorchVAEImpl` or `PyroVAEImpl`) based on the `args.impl` parameter.
    - Static dispatch happens during the initialization of optimizers, where method overloading or polymorphism selects the optimizer based on the VAE implementation.

5. **Error Handling (Lines 152-169):** 
    - Error handling is employed through the use of assertions (`assert`) to ensure the correct version of Pyro is being used and to handle incorrect input for the chosen implementation.

6. **Concurrency and Parallelism (Not explicitly present):** 
    - While not explicitly coded in this implementation, the code structure doesn't directly employ concurrency or parallelism concepts. Utilizing PyTorch's or Pyro's capabilities in a distributed computing environment could incorporate these concepts.

7. **Memory Management and Resource Handling (Lines 128, 147-151):** 
    - Memory management is crucial in the optimization step (`self.optimizer.step()`), where gradients are computed and parameters are updated. Also, resource handling for file operations (`os.makedirs()`) and clearing parameter stores (`pyro.clear_param_store()`) is essential.

8. **Control Structures (Lines 150-170, 175-178):** 
    - The code uses control structures (`if-else`, `for`) for conditional execution (checking implementation type, deciding on skipping evaluation, etc.) and iterative tasks (training loop).

9. **State and Mutation (Lines 30-37, 56-60, 111-113):** 
    - State and mutation are managed throughout the code, notably with the change of mode (`TRAIN` or `TEST`) affecting the behavior of the VAE (`self.mode = TRAIN/TEST`), and through the optimization process where parameters are updated (`self.optimizer.step()`).

10. **Libraries and Frameworks (Lines 7-14, 25, 27-28, 34, 51, 91-98):** 
    - The code extensively uses external libraries (PyTorch, Pyro) and frameworks (torch.nn, pyro.infer, torchvision, etc.) that offer high-level functionalities to build and train neural networks and probabilistic models, abstracting low-level implementation details.
   
* Regarding difficulties faced, integrating Pyro and PyTorch into the VAE implementation presented challenges in terms of different APIs, debugging, and aligning the functionality between the two frameworks.
* Managing the state and mode changes to be error-prone was also one of the challenges faced.
* Additionally, ensuring compatibility and consistency between the different versions of the libraries was a concern.
---
## Results

> Tests conducted. Dataset used. Benchmarks run. Show graphs. Line graphs, bar graphs, etc. How are you checking/validating that these results align with your initial problem statement. Data-driven proof points that the solution/system is working. Why should I be convinced it is working?

#### Tests and Results on MNIST Dataset
Results can be found in the `results` folder.

#### Problem Statement Alignment:

Both implementations aim to address image reconstruction and latent space representation learning using different frameworks, PyTorch and Pyro. The common objective is to reconstruct MNIST digits and potentially generalize on unseen data.

#### Validation Strategies:

**PyTorch Implementation:**
- **Image Reconstruction:** Employs standard PyTorch modules for the encoder and decoder. Uses PyTorch's `torch.optim.Adam` for optimization.
- **Loss Functions:** Utilizes binary cross-entropy and KL divergence to ensure accurate image reconstruction and meaningful latent representations.
- **Evaluation Metrics:** Computes training and test losses per epoch to monitor model performance.

**Pyro Implementation:**
- **Probabilistic Programming:** Utilizes Pyro for modeling and inference via probabilistic programming.
- **Custom Model & Guide:** Defines custom probabilistic models for the encoder and decoder.
- **Elbo Loss Optimization:** Uses SVI (Stochastic Variational Inference) with Elbo loss for optimization.

#### Validation Approaches:

**PyTorch Implementation:**
- **Validation:** Compares reconstructed images to originals and analyzes loss trends over training epochs. Checks for generalization on test data.

**Pyro Implementation:**
- **Probabilistic Validation:** Relies on Pyro's probabilistic modeling for validation. Evaluates performance using SVI and Elbo loss convergence.

#### Performance Validation:

**PyTorch Implementation:**
- **Strengths:** Utilizes PyTorch's flexibility and standard optimization tools for VAE implementation.
- **Considerations:** May require additional manual handling for certain probabilistic aspects compared to Pyro.

**Pyro Implementation:**
- **Strengths:** Leverages Pyro's probabilistic programming for a more expressive probabilistic model.
- **Considerations:** Could have a steeper learning curve due to probabilistic programming concepts.

### Conclusion:

Even though both PyTorch and Pyro implementations align with the problem statement of VAE-based image reconstruction and latent space learning, PyTorch provides a more straightforward, traditional approach, while Pyro leverages the power of probabilistic programming for a more flexible and expressive modeling process.

---
## Potential for future work

> If you were given more time what else would you do. What other POPL aspects might come into play?

1. **Advanced Generative Models**
- Continued exploration and development of new generative models beyond VAEs, such as improvements to Generative Adversarial Networks (GANs), flow-based models, and hybrid architectures that combine the strengths of different models.

2. **Enhanced Sample Quality and Diversity**
- Improving the quality and diversity of generated samples, addressing challenges such as mode collapse in GANs and finding solutions for generating more realistic and diverse outputs in various domains.

3. **Efficient Training Algorithms**
- Designing more efficient training algorithms for generative models, including advancements in optimization techniques, regularization methods, and strategies for handling large-scale datasets.

4. **Uncertainty Modeling**
- Advancements in uncertainty modeling, providing generative models with better capabilities to quantify and express uncertainty in predictions, which is crucial for real-world applications such as medical diagnosis and decision-making.

5. **Some other POPL Aspects involved**
- Concurrency and Parallelism
- Type Systems
- Domain-Specific Languages (DSLs)
  
---

