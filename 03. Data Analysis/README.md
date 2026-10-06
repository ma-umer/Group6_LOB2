# Data Analysis
Before training any prediction method, we first look at our data. This step answers simple questions: How long are our proteins? How long are the signal peptides? Which amino acids do they contain? Which organisms do they come from? And what does the cleavage site look like?

Looking at the data first helps us:

check that the training and benchmark sets are similar (so the final test is fair),
spot biases in the data,
understand which features could help a model recognise a signal peptide (SP).

Script: [data_analysis.py](https://github.com/ma-umer/Group6_LOB2/blob/main/03.%20Data%20Analysis/data_analysis.py)

Output: [Plots](https://github.com/ma-umer/Group6_LOB2/tree/main/03.%20Data%20Analysis/plots)

## The data we analysed

The two datasets come from `02. Data Preparation`. Each one is described
**separately**, as asked in the course.

| Dataset | Proteins with SP (positives) | Proteins without SP (negatives) | Total |
|---|---:|---:|---:|
| Training | 876 | 7,265 | 8,141 |
| Benchmark | 219 | 1,817 | 2,036 |

> **Quick reminder:** a *signal peptide* is a short "address label"
> (about 20–30 amino acids) at the start (N-terminus) of a protein.
> It sends the protein into the secretory pathway and is then cut off at
> the *cleavage site*. It has three parts: a positively charged
> **n-region**, a hydrophobic (water-avoiding) **h-region**, and a
> **c-region** that contains the cleavage site.

## 1. Protein length
![Protein Length Comparison Plot](./03_Data_Analysis/plots/protein_length_comparison.png)
