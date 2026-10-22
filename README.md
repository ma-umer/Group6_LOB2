# Group6_LOB2 🧬✂️

> ⚠️ **Disclaimer**
>
> This repository contains student work. It was written by students,
> not by a team of professional researchers locked in a basement with
> unlimited compute :)
>
> Expect experiments, mistakes, questionable code, things that probably
> could have been done better, and the occasional piece of code that
> somehow works. The project is primarily for learning and educational
> purposes.

---

Some proteins are made to leave the cell. To get out, they carry a short tag at their front, called a **signal peptide**: a kind of boarding pass that the cell checks and then cuts off. Rude, but efficient.

**Our goal:** teach a computer to look at a protein sequence and tell whether it has that boarding pass.

This is the **Laboratory of Bioinformatics 2** project (University of Bologna, A.Y. 2026–2027) by **Group 6**.

---

## The project in short

We collect thousands of eukaryotic proteins from **UniProt**, some with a signal peptide and some without. We clean the data, explore it, and then build two predictors:

- **von Heijne method:** a classic approach that learns the typical amino-acid pattern around the signal-peptide cut site.
- **SVM (Support Vector Machine):** a machine-learning model trained on features calculated from each protein.

Finally, we test both methods on proteins they have never seen and compare which one works better.

---

## Project steps

| Step | What we do | Status |
|---|---|---|
| [01 · Data collection](01_data_collection) | Download proteins with and without a signal peptide | ✅ Done |
| [02 · Data preparation](02_data_preparation) | Remove similar sequences, split into training and benchmark sets | ✅ Done |
| [03 · Data analysis](03_data_analysis) | Explore and visualise the data | ✅ Done |
| [04 · von Heijne](04_von_heijne) | Weight-matrix predictor | ✅ Done |
| [05 · SVM](05_svm) | Machine-learning predictor | 🔨 In progress |
| Report | Compare both methods and write it all up | ⏳ To do |

Each folder has its own README with the details.

---

## Results so far

| Method | Cross-validation MCC | Benchmark MCC |
|---|---|---|
| von Heijne | 0.675 ± 0.011 | 0.683 |
| SVM | 0.86 ± 0.01 | 0.87 |

MCC goes from 0 (random guessing) to 1 (perfect). The SVM wins clearly, mostly because it gets fooled far less often by transmembrane proteins, which look a lot like signal peptides. The old method had a point, though: it did all that with a single table of numbers.

---

## Tools

Python (pandas, NumPy, matplotlib, seaborn, scikit-learn), UniProt, MMseqs2, Google Colab, and a worrying amount of coffee ☕

---

## Team

| Name | GitHub |
|---|---|
| *Ahmed Abdelhafize* | [@username](https://github.com/username) |
| *Berkay Oktay Demir* | [@username](https://github.com/username) |
| *Mohammad Amin Roshan* | [@username](https://github.com/username) |
| *Mohibuddin Ahmed Umer* | [@username](https://github.com/username) |
| *Rida Niaz* | [@username](https://github.com/username) |

**Course:** Laboratory of Bioinformatics 2, MSc in Bioinformatics, University of Bologna
**Instructor:** Prof. Castrense Savojardo

---

*No proteins were harmed in the making of this project. Several signal peptides were, however, cleaved.* ✂️
