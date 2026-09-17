## LAB-2 PROJECT DATA COLLECTION

# Selection Criteria For The Positive Set of Proteins

# Selection Criteria For the Negative Set of Proteins

#UniProtKB 2,972 results For Positive Query

Positive Query : (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)

#UniProtKB 20,975 results For Negative Query

Negative Query : (reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (existence:1) NOT (ft_signal:*) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039) OR (cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0173)	

#Positve query API URL: https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28reviewed%3Atrue%29+AND+%28ft_signal_exp%3A*%29%29&size=500

#Format:JSON, Compressed:No

Negative query API URL: https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND%28reviewed%3Atrue%29+AND+%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28existence%3A1%29+NOT+%28ft_signal%3A*%29+OR+%28cc_scl_term_exp%3ASL-0191%29+OR+%28cc_scl_term_exp%3ASL-0204%29+OR+%28cc_scl_term_exp%3ASL-0039%29+OR+%28cc_scl_term_exp%3ASL-0091%29+OR+%28cc_scl_term_exp%3ASL-0209%29+OR+%28cc_scl_term_exp%3ASL-0173%29%29&size=500

#Format:JSON, Compressed:No


Script:
