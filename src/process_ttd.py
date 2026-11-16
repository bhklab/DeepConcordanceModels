import sys
import pandas as pd
import utils



def process_block(txt,res_dict,drug_list):

	all_drugs = []
	# print("****MY TEXT****")
	# print(txt)
	is_gene_target = False
	in_drug_universe = False
	for line in txt:
		data = line.rstrip().split("\t")
		
		if data[1] in ['GENENAME','DRUGINFO']:
			if data[1]== 'GENENAME':
				gene = data[2]
				is_gene_target = True
			elif data[1] == "DRUGINFO":
				drug_names = [x.upper() for x in data[3].split("/")]
				# print(drug_list)
				# print(drug_names)
				if any([x in drug_list for x in drug_names]):
					in_drug_universe = True
				# print([x.lower() for x in drug_names])
				# if len(drug_names)>1:
					# print(drug_list)
					# sys.exit()
				# in_drug_universe = any()
				all_drugs.extend(drug_names)


	
	if is_gene_target and in_drug_universe:
		res_dict[gene] = all_drugs
	# print(res_dict)
drugs = utils.read_text_file("../procdata/drug_lists/common_drugs.txt")
drugs = [x.upper() for x in drugs]

with open("../rawdata/P1-01-TTD_target_download.txt","r") as istream:
	lines = istream.readlines()

 # text file header stops here

lines = lines[32:]
curr = 0
target_to_drugs = {}
while(curr<len(lines)):
	nxt = curr
	while nxt<len(lines) and lines[nxt]!="\n":
		nxt+=1
	block = lines[curr:nxt]
	# print(block)
	process_block(block,target_to_drugs,drugs)
	curr = nxt+1
	# print(block)
	print(curr)
	# print(nxt)

targets = [k if k!='Candi TMP1' else 'TPM1' for  k in target_to_drugs.keys() ]
with open("../procdata/gene_sets/ttd_targets.txt","w") as ostream:
	ostream.writelines([x+"\n" for x in targets])

# target_to_drugs = {k:target_to_drugs[k] for k in target_to_drugs.keys() if }