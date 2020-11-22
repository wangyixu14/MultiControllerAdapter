# Automatic Neural Network Controller Design for Learning-Enabled Cyber-Physical Systems
This repo is targeting the research of towards safe, efficient, robust, and verifiable neural network controller synthesis for LE-CPSs based on multiple available control experts.  

## Examples
1. Van der Pol's Oscillator
2. 3-dimensional polynomial system 
3. Cartpole system

## Primal Results
```console
~$ cd os_ppo
~$ python adaptation.py
```
Similar commands apply to /3d_ppo/, /cartpole_ppo/
<!-- ## ICCAD 2020: Energy-Efficient Control Adaptation With Safety Guarantees for Learning-Enabled Cyber-Physical Systems
The key code of our ICCAD 2020 paper is contained in this repo. Please check the ./os_ppo/ subfolder. 
1. The MATLAB code is used to compute the inner-approximation of robust invariant set for the synthsized robust neural network controller. 
2. The switching strategy learning process by Double DQN is also shown in the adaptation.py.   -->

<!-- ## Reference:
To cite this paper:
@INPROCEEDINGS{yixuan2020,
  title={Energy-Efficient Control Adaptation with Safety Guarantees for Learning-Enabled Cyber-Physical Systems},
  author={Wang, Yixuan and Huang, Chao and Zhu, Qi},
  booktitle={International Conference on Computer-Aided Design (ICCAD)},
  year={2020},
  doi={10.1145/3400302.3415676}
} -->
