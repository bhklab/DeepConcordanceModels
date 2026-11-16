###
#
#
#  Function to compute assay noise in an assay
#	
#	iterates over all biological replicates and computes AAC gap between them
# 
#
#
####

compute.assay.noise <- function(pset, viability.col= "aac_recomputed"){
	rep.counts <- sensNumber(pset)
	replicates <- which(rep.counts>1,arr.ind=TRUE,useNames = FALSE)

	num.replicated <- dim(replicates)[1]
	
	info <- treatmentResponse(pset)$info
	responses <- treatmentResponse(pset)$profiles
	gaps <- c()
	xs <- c()
	ys <- c()
	samples <- c()
	drugs <- c()

	for(i in 1:nrow(replicates)){

		sample.idx <- replicates[i,1]
		drug.idx <- replicates[i,2]
		
		
		sample <- rownames(rep.counts)[sample.idx]
		drug <- colnames(rep.counts)[drug.idx]
		
		# names of replicated experiments
		exp.names <- info %>% 
			filter(sampleid==sample) %>% 
			filter(treatmentid == drug) %>%
			rownames()

		replicate.viabilities <- responses[exp.names, viability.col]
		# num replicates for this sample and drug pair
		sample.drug.replicates <- length(replicate.viabilities)
		
		

		# num replicates choose 2
		# gives all pairs of viabilities to compare deltas
		points <- t(combn(replicate.viabilities,2))
		samples <- c(samples, rep(sample,dim(points)[1]))
		drugs <- c(drugs,  rep(drug,dim(points)[1]))

		# add absolute value of gaps
		points <- cbind(points, abs(points[,1] - points[,2]))
		
		# add to gaps
		xs <- c(xs, points[,1])
		ys <- c(ys, points[,2])
		gaps <- c(gaps, points[,3])
	}
	
	results = data.frame(Drug = drugs, CCL = samples, X.Coord = xs, Y.Coord = ys, Delta = gaps)
	return(results)
}
