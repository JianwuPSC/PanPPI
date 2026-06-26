import sys
import gzip
import json

def load_local_uniprot_data(path):
    total_dict={}
    with gzip.open(path, 'rt') as f:
        total = json.load(f)['results']
    for entry1 in total:
        if "primaryAccession" in entry1:
            total_dict[entry1["primaryAccession"]] = entry1
    return total_dict

def get_local_location(uniprot_id, local_db):
    if uniprot_id in list(local_db.keys()):
        entry = local_db.get(uniprot_id, {})
        locations = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                locations.extend(
                    loc["location"]["value"]
                    for loc in comment.get("subcellularLocations", [])
                )
        return list(set(locations))
    else:
        return []

def get_fasta_dict(path_a):    
    fa_dict = {}  # fasta seqs
    with open(path_a) as fa:
        for line in fa:
            line = line.replace('\n','')
            if not line.startswith('#'):
                if line.startswith('>'):
                    seq_name = line[1:]
                    fa_dict[seq_name] = ''
                else:
                    fa_dict[seq_name] += line.replace('\n','')
    return fa_dict


path_json = sys.argv[1]
path_fasta = sys.argv[2]
output_path = sys.argv[3]

local_db = load_local_uniprot_data(path_json)
fa_dict = get_fasta_dict(path_fasta)

sub_location_dict={}
for i in list(local_db.keys()):
    string_name = str()
    if 'uniProtKBCrossReferences' in local_db[i]:
        for aa in local_db[i]['uniProtKBCrossReferences']:
            if 'STRING' in aa['database']:
                string_name = aa['id']
    if string_name == str():
        string_name = '.'.join([str(local_db[i]['organism']['taxonId']),i])
    sub_location_dict[string_name] = get_local_location(str(i),local_db)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sub_location_dict, f, indent=4)  # indent
