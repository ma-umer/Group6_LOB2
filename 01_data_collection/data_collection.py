
import re
import json
import requests
from requests.adapters import HTTPAdapter, Retry


url_positive = "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28reviewed%3Atrue%29+AND+%28ft_signal_exp%3A*%29%29&size=500"
url_negative = "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND%28reviewed%3Atrue%29+AND+%28fragment%3Afalse%29+AND+%28taxonomy_id%3A2759%29+AND+%28length%3A%5B40+TO+*%5D%29+AND+%28existence%3A1%29+NOT+%28ft_signal%3A*%29+OR+%28cc_scl_term_exp%3ASL-0191%29+OR+%28cc_scl_term_exp%3ASL-0204%29+OR+%28cc_scl_term_exp%3ASL-0039%29+OR+%28cc_scl_term_exp%3ASL-0091%29+OR+%28cc_scl_term_exp%3ASL-0209%29+OR+%28cc_scl_term_exp%3ASL-0173%29%29&size=500"
column_positive =["Accession", "Organism", "Kingdom", "Length", "SP cleavage"]
column_negative =["Accession", "Organism", "Kingdom", "Length", "N-term TM"]

kingdoms = ["Metazoa", "Viridiplantae", "Fungi"]

next_link_re = re.compile(r'<(.+)>; rel="next"')

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])))

def iter_entries(url):
    while url:
        resp = session.get(url)
        resp.raise_for_status()
        yield from resp.json()["results"]
        match = next_link_re.search(resp.headers.get("Link", ""))
        url = match.group(1) if match else None



def get_kingdom(entry):
    lineage = entry["organism"]["lineage"]
    return next((k for k in kingdoms if k in lineage), "Other")


def has_signal_peptide(entry, min_length=14):
    features = entry.get("features") or []
    if not features or features[0]["type"] != "Signal":
        return False
    feat = features[0]
    if feat["description"]:
        return False
    return feat["location"]["end"]["value"] > min_length

def has_nterm_transmembrane(entry, window=90):
    for feat in entry.get("features", []):
        if (feat["type"]== "Transmembrane") and "Helical" in feat.get("description", "") and feat["location"]["start"]["value"] <= window:
            return True
    return False


def write_dataset(entries, columns, row_fn, tsv_path, fasta_path):
    total = kept = 0
    with open(tsv_path, "w") as tsv, open(fasta_path, "w") as fasta:
        tsv.write("\t".join(columns) + "\n")
        for entry in entries:
            total += 1
            row = row_fn(entry)
            if row is None:
                continue
            kept += 1
            tsv.write("\t".join(str(v) for v in row)+ "\n")
            fasta.write(f">{entry['primaryAccession']}\n{entry['sequence']['value']}\n")
    print(f" Total :{total} | Kept:{kept}")
    return total, kept




def positive_row(entry):
    if not has_signal_peptide(entry):
        return None
    return (
        entry["primaryAccession"],
        entry["organism"]["scientificName"],
        get_kingdom(entry),
        entry["sequence"]["length"],
        entry["features"][0]["location"]["end"]["value"],
    )

def negative_row(entry):
    return (entry["primaryAccession"], entry["organism"]["scientificName"], get_kingdom(entry), entry["sequence"]["length"], has_nterm_transmembrane(entry))



if __name__ == "__main__":
    print("Positive entries:")
    write_dataset(iter_entries(url_positive), column_positive, positive_row,
                  "positive.tsv", "positive.fasta")

    print("\nNegative entries:")
    write_dataset(iter_entries(url_negative), column_negative, negative_row,
                  "negative.tsv", "negative.fasta")
