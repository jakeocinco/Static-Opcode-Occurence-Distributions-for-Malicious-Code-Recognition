# Static Opcode Occurence Distributions for Malicious Code Recognition

The data and code used for 'Experiments on Recognition of Malware based on Static Opcode Occurrence Distributions' and University of Cincinati MS Thesis, written by Jacob Carlson and Anca Ralescu

## Abstract

This work discusses a static method for recognizing malicious code samples by comparing opcode distribu- tions created through a novel approach. Distributions are created by aggregating the number of operations between consecutive calls of an opcode. Creating these distributions for each file and comparing them to ground truth distributions representing benign and malicious code samples creates a set of input values that can lead to an accurate method of malicious code detection. This paper also provides a dataset of distributions, as well as describes the methods used to create these distributions.

## Data
Data includes ground truth distributions and valdiation accuracies from the training set.
```
├── op_code_distributions_samples
│   ├── distribution_method_1 
│   │   ├── 'pruned' or 'base'
│   │   │   ├── opcode set 1 
│   │   │   │   ├── linear
│   │   │   │   │   ├── #_bins 
│   │   │   │   │   │   ├── #_samples
│   │   │   │   │   │   │   ├── 'clean' or 'infected'
│   │   │   │   │   │   │   │   ├── #.npy 
│   │   │   ├── opcode set 2 
│   ├── distribution_method_2
├── results
│   ├── {method}_{random_seed}.json
```
