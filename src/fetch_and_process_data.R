###
#
#	Code to fetch pharmacogenomics data 
#	from the ORCESTRA.ca database and parse 
#	it in into .csv files.
#
#
#
#
#		Generates screen-specific csvs with drug data, drug response data,
# 		and harmonizes the data into. 
#		Also generates the files with replicates. 
#
#
#
#	Author: James Bannon
#	Github: jbannon
#

# load libraries
suppressWarnings(library(PharmacoGx))
suppressWarnings(library(dplyr))
suppressWarnings(library(tidyr))
suppressWarnings(library(curl))

source("compute.assay.noise.R")

############ Parameters #########


# Directories
proc.dir <- "../procdata" # Destination for processed data files
raw.dir <- "../rawdata"   # Where to keep the raw data

liquid.tumor.types <- c("Myeloid","Lymphoid") # We will keep track of non-liquid ccls

## A list where the entry names correspond to the datset
## and the values are pairs of the form (<download_link>,<file_name>).

download.info <- list(
		CCLE = c("https://zenodo.org/record/3905462/files/CCLE.rds?download=1","CCLE.rds"),
		CTRPv2 = c("https://zenodo.org/record/7826870/files/PSet_CTRPv2.rds?download=1","PSet_CTRPv2.rds"),
		GDSC2 =  c("https://zenodo.org/records/5787145/files/GDSC2.rds?download=1","GDSC2.rds"),
		GDSC1 =  c("https://zenodo.org/records/7829915/files/PSet_GDSC2020.rds?download=1","PSet_GDSC2020.rds"),
		gCSI = c("https://zenodo.org/record/7829857/files/PSet_gCSI2019.rds?download=1","PSet_gCSI2019.rds")
		)

## A list where the entry names correspond to the shorthand names for
## the screens used and values are tuples of the form (<data_file>,<ccl_stem>) 
## with the exception of Broad which is of the form 
## (<expression_file>,<response_file>,<ccl_stem>).
## The ccl_stem is how the specify datasets label their cell lines.

data.info <- list(
	gCSI = c("PSet_gCSI2019.rds","EGAN"),
	GDSC2 = c("GDSC2.rds","EGAR"),
	GDSC1 = c("PSet_GDSC2020.rds","EGAR"),
	Broad = c("CCLE.rds","PSet_CTRPv2.rds","SRR")
	)


### BEGIN PROCESSING


###
#
#	Fetching loop
#
####
cat("\n\n******* Starting Fetch Loop *******\n\n")
for(data.set in names(download.info)){
	cat(paste0("Checking ",data.set,"..."))
	download.link = download.info[[data.set]][1]
	file.name = download.info[[data.set]][2]

	if(file.exists(file.path(raw.dir,"psets",file.name))){
		cat("\n\t\t...local copy detected!\n")
	} else{
		cat("\n\t\t...downloading!")
		curl_download(download.link,file.path(raw.dir,"psets",file.name),quiet=FALSE)
	}
	cat("\n")
}

cat("\n")
###
#
#	Process downloaded files
#
###

cat("\n\n******* Starting Processing Loop *******\n\n")


## Iterate over the downloaded datasets
for(data.set in names(data.info)) {


	cat(paste0("\nProcessing ",data.set,"\n"))

	write.dir <- file.path(proc.dir,data.set,"PSet_Data","")
	if(!dir.exists(write.dir)){
	dir.create(write.dir)
	}

	ccl.stem <- tail(data.info[[data.set]],1)
	
	
	
	if(data.set %in% c("gCSI","GDSC2","GDSC1")){

		## For gCSI,GDSC2, and GDSC1 the expression and response info are in the same
		## PSet. 
	
		PSet.file <- file.path(raw.dir,"psets",data.info[[data.set]][1])
		print(PSet.file)
		ps <- readRDS(PSet.file)
		ps <- updateObject(ps)
		exp.pset <- ps 
		resp.pset <- ps

	} else {
		
		## For the Broad screens we need to use the CCLE PSet
		## for expression and the CTPRv2 PSet for drug response.

		PSet.file.expression <- file.path(raw.dir,"psets", data.info[[data.set]][1])
		PSet.file.response <- file.path(raw.dir,"psets", data.info[[data.set]][2])
		
		resp.pset <- readRDS(PSet.file.response)
		exp.pset <- readRDS(PSet.file.expression)
		resp.pset <-  updateObject(resp.pset)
		exp.pset <- updateObject(exp.pset)

	}

	## pull the info about the drugs used in this screen into a dataframe
	## keep the SMILES strings for potential later use.

	results <- compute.assay.noise(resp.pset)
	
	if(!dir.exists("../results/qc/")){
		dir.create("../results/qc/")
	}

	write.csv(results,paste0("../results/qc/",data.set,"_replicates.csv"))

	drug.data <- as.data.frame(drugInfo(resp.pset))%>% select(treatmentid,smiles)
	rownames(drug.data) <- NULL

	drug.names <- drug.data %>% pull(treatmentid) %>% unique()


	## Put the CCL metadata into a data frame
	ccl.info <- as.data.frame(sampleInfo(exp.pset))
	
	

	liquid.ccls <- ccl.info %>% 
		filter(tissueid %in% liquid.tumor.types) %>% 
		pull(Cellosaurus.Accession.id)
	

	ostream <-file(file.path(write.dir,"liquid_ccls.txt"))
	writeLines(liquid.ccls,ostream)
	close(ostream)


	solid.ccls <- ccl.info %>%
		filter(!tissueid %in% liquid.tumor.types) %>%
		pull(Cellosaurus.Accession.id)

	ostream <- file(file.path(write.dir,"solid_ccls.txt"))
	writeLines(solid.ccls,ostream)
	close(ostream)


	all.ccls <- ccl.info %>% pull(Cellosaurus.Accession.id)

	all.ccl.sampleids <- ccl.info %>%pull(sampleid)

	ostream <- file(file.path(write.dir,"all_ccls.txt"))
	writeLines(all.ccls,ostream)
	close(ostream)
	

	# Select only the sample ids and the tissue
	# the ccl data and drug data
	ccl.info <- ccl.info %>%
		select(sampleid, Cellosaurus.Accession.id,  tissueid)

	write.csv(ccl.info,file.path(write.dir,"ccl_data.csv"),row.names=FALSE)
	write.csv(drug.data,file.path(write.dir,"drug_data.csv"),row.names=FALSE)

	# pull the treatment response info
	experiment.info <- as.data.frame(treatmentResponse(resp.pset)$info)
	
	experiment.info <- experiment.info %>% filter(sampleid %in% all.ccl.sampleids)
	
	response.vars <- treatmentResponse(resp.pset)$profiles %>% 
		select(aac_recomputed) %>% 
		drop_na()




	## Summarize the response variables by
	## averaging the aac response across replicates

	responses <- merge(experiment.info,response.vars, by=0) %>% 
		select(sampleid, treatmentid, aac_recomputed) %>% 
		group_by(sampleid,treatmentid) %>%
		summarize(aac = mean(aac_recomputed))
	
	cat("\n\nBefore selecting for Cellosaurus ID\n")
	cat(dim(responses))
	responses <- merge(responses, ccl.info,by="sampleid") %>% select(Cellosaurus.Accession.id,treatmentid, aac)
	cat("\n\nAfter selecting for Cellosaurus ID\n")
	cat(dim(responses))
	## Rename the response data frame columns and write it
	colnames(responses) <-c("Cell Line",'Drug','AAC')
	write.csv(as.data.frame(responses),file.path(write.dir,"responses.csv"),row.names=FALSE)
	
	## pull expression data
	expression <- molecularProfiles(exp.pset)$Kallisto_0.46.1.rnaseq

	## get meta data about the measured genes for filtering
	gene.info <- as.data.frame(rowData(expression)) %>% 
		select(gene_name) 
	

	## List all the unique genes measured
	all.genes <- as.vector(gene.info$gene_name) %>% unique()
	
	## Pull out only the protein coding genes
	protein.coding.genes <- as.data.frame(rowData(expression)) %>%
		filter(gene_type == 'protein_coding') %>%
		select(gene_name)

	
	
	pc.gene.list <- as.vector(protein.coding.genes$gene_name) %>% unique()


	## Write these gene lists to files
	ostream <- file(file.path(write.dir,"pc_genes.txt"))
	writeLines(pc.gene.list,ostream)
	close(ostream)

	ostream <- file(file.path(write.dir,"all_genes.txt"))
	writeLines(all.genes,ostream)
	close(ostream)
 
 	## create an expression data frame using just the protein coding genes
 	## we use the ccl.stem to match the gene-id 
 	## and take the median of expression values in the case that there are multiple
 	## measurements for one gene.
 	
	expression.data <- merge(protein.coding.genes, as.data.frame(assay(expression)), by=0)%>%
		select(-Row.names) %>%
		group_by(gene_name) %>% 
		summarise_at(vars(matches(ccl.stem)),median)

	
	keep.ccls <- as.data.frame(colData(expression)) %>%	
			filter(sampleid %in% all.ccl.sampleids) %>% 
			select(sampleid)

	
	
	colnames(keep.ccls)<-c("Cell_line")
	
	expression.data <- expression.data %>% 
		as.data.frame() %>% 
		tibble::column_to_rownames("gene_name") %>% t()


	expression.data <- keep.ccls %>% 
		select(Cell_line) %>% 
		merge(expression.data, by=0) %>% 
		select(-Row.names) %>% 
		group_by(Cell_line) %>% 
		summarize_all(median) %>%
		as.data.frame()


	expression.data <- merge(expression.data,ccl.info,by.x="Cell_line",by.y="sampleid") %>%
		select(-c(Cell_line, tissueid)) %>% select(Cellosaurus.Accession.id,everything())
	

	colnames(expression.data)[1] <- "Cell Line"
	print
	write.csv(expression.data,paste0(write.dir,"/expression.csv"),row.names=FALSE)
}

cat("\n\n******* Computing Intersection of Measured Genes *******\n\n")
pc.list = list()
all.gene.list = list()
for(data.set in names(data.info)){
	pc.file <- file.path(proc.dir,data.set,"PSet_Data","pc_genes.txt")
	all.gene.file <- file.path(proc.dir,data.set,"PSet_Data","all_genes.txt")

	pc.temp <-  read.delim(pc.file, header = FALSE)
	pc.list[[data.set]] <- pc.temp$V1

	all.temp <- read.delim(all.gene.file, header=FALSE)
	all.gene.list[[data.set]] <- all.temp$V1
}


common.pcs <- Reduce(intersect, list(pc.list$Broad, pc.list$gCSI, pc.list$GDSC2))

#all_genes <- Reduce(intersect, list(all.gene.list$Broad, all.gene.list$gCSI, all.gene.list$GDSC2))
#all_genes <-  as.vector(intersect(broad_genes$V1,intersect(gdsc_genes$V1,gCSI_genes$V1)))

ostream <- file(file.path(proc.dir,"common_pc_genes.txt"))
writeLines(common.pcs,ostream)
close(ostream)

## need to change character because the - gets
## replaced with a . when writing 
common.pcs <- gsub("-",".",common.pcs)



cat("\n\n******* Subsetting Expression to Common Protein Coding Genes *******\n\n")
for(data.set in names(data.info)){
	
	expr <- read.csv(file.path(proc.dir,data.set,"PSet_Data","expression.csv"))
	expr <- expr %>% select(all_of(c('Cell.Line', common.pcs)))
	colnames(expr)[1]<- "Cell Line"

	write.csv(expr,file.path(proc.dir,data.set,"PSet_Data","expression.csv"),row.names=FALSE)
}


cat("\n\n******* Processing Gene Sets *******\n\n")

gs.dir <- "../rawdata/gene_sets"

gs.files <- list(KEGG = "kegg.txt",
				COSMIC = "COSMIC.txt",
				LINCS = "LINCS.txt",
				MDSIG = "mdsig_hallmarks.txt")

write.dir <- file.path(proc.dir,"gene_sets","")

if(!dir.exists(write.dir)){
	dir.create(write.dir)
}

all.genes <- c()


for(i in seq(length(gs.files))) {
	
	gs1.name <- names(gs.files)[i]
	gs1.file <- file.path(gs.dir,gs.files[[gs1.name]])
	
	gs1 <- read.delim(gs1.file,header=FALSE)
	ostream <- file(file.path(write.dir,paste0(gs1.name,".txt")))
	writeLines(gs1$V1,ostream)
	close(ostream)

	all.genes <-c(all.genes,gs1$V1)
	
	for(j in seq(i,length(gs.files))) {
		
		if (i!=j){
			
			gs2.name <- names(gs.files)[j]
			
			gs2.file <- file.path(gs.dir,gs.files[[gs2.name]])
			gs2 <- read.delim(gs2.file,header=FALSE)

			union.gs <- union(gs1$V1, gs2$V1)
			ostream <- file(file.path(write.dir,paste0(gs1.name,"+",gs2.name,".txt")))
			writeLines(union.gs,ostream)
			close(ostream)
		}
	}
}

all.genes <- unique(all.genes)
ostream <- file(file.path(write.dir,"all_genesets.txt"))
writeLines(all.genes,ostream)
close(ostream)
cat("\n\n All Done!\n\n")