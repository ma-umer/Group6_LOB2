# Group6_LOB2
> ⚠️ **Disclaimer**
>
> This repository contains students work. It was written by students,
> not by a team of professional researchers locked in a basement with
> unlimited compute.
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

**Course:** Laboratory of Bioinformatics 2, MSc in Bioinformatics, University of Bologna
**Instructor:** Prof. Castrense Savojardo

---

*No proteins were harmed in the making of this project. Several signal peptides were, however, cleaved.* ✂️

