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
python basevae.py --impl pyro
```

Running the code for Pytorch
```bash
python basevae.py --impl pytorch
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

* **Memory Safety**: The use of `torch.tensor` and `torch.nn` modules from the pytorch library, along with the ownership system and borrowing rules, contributes to memory safety.

---
## Results

> Tests conducted. Dataset used. Benchmarks run. Show graphs. Line graphs, bar graphs, etc. How are you checking/validating that these results align with your initial problem statement. Data-driven proof points that the solution/system is working. Why should I be convinced it is working?


---
## Potential for future work

> If you were given more time what else would you do. What other POPL aspects might come into play?


---

