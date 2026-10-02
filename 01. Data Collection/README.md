# LAB-2 PROJECT DATA COLLECTION

## Selection Criteria For The Positive Set of Proteins
The URL for Positive dataset was generated from the UniProtKB Advanced search. It selects: reviewed, eukaryotic, non-fragment proteins, length ≥ 40, evidence at protein level, with a signal peptide supported by experimental evidence.

The remaining criteria cannot be expressed in the query, so we check them with a filter function:
1. the signal peptide has experimental evidence (ECO:0000269)
2. the signal peptide is at least 14 residues long

## Selection Criteria For the Negative Set of Proteins
The URL for negative dataset selects the same common criteria as the positive, and in addition:
* no signal peptide annotated at any evidence level (`NOT ft_signal:*`)
* experimental subcellular location in cytosol (SL-0091), nucleus (SL-0191), mitochondrion (SL-0173), plastid (SL-0209), peroxisome (SL-0204) or cell membrane (SL-0039)

## Positive Query 

### UniProt Query For Positive Proteins
```text
 (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)
```
***UniProtKB 2,972 results***

### Positive query API URL:
```
https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28reviewed%3Atrue%29+AND+%28ft_signal_exp%3A*%29%29&size=500 #
```
***Format:JSON, Compressed:No***



## Negative Query 

### UniProt Query For Negative Proteins
```
(reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (existence:1) NOT (ft_signal:*) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039) OR (cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0173)	
```
***UniProtKB 20,975 results***


### Negative query API URL: 
***Format:JSON, Compressed:No***

```
https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND%28reviewed%3Atrue%29+AND+%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28existence%3A1%29+NOT+%28ft_signal%3A*%29+OR+%28cc_scl_term_exp%3ASL-0191%29+OR+%28cc_scl_term_exp%3ASL-0204%29+OR+%28cc_scl_term_exp%3ASL-0039%29+OR+%28cc_scl_term_exp%3ASL-0091%29+OR+%28cc_scl_term_exp%3ASL-0209%29+OR+%28cc_scl_term_exp%3ASL-0173%29%29&size=500
```
### Data Collection:
*** [data_collection.py](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/data_collection.py)
## DATA SUMMARY
---
|Data Set| Total Proteins|Filtered Proteins|Transmembrane Proteins|Files|
|--------|---------------|-----------|--------|----------------|
|Positive|  2972         |    2953   |    -   | [FASTA](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/positives/positive.fasta)  [TSV](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/positives/positive.tsv) |
|Negative|  20975        |    20975  |   2531 | [FASTA](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/negatives/negative.fasta)  [TSV](https://github.com/ma-umer/Group6_LOB2/blob/main/01.%20Data%20Collection/negatives/negative.tsv) |
---
