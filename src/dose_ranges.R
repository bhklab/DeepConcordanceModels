library(PharmacoGx)
library(dplyr)

drugs <- c('5-Fluorouracil',"AZD7762","AZD8055",
        "Bortezomib","Crizotinib","Dasatinib",
        "Erlotinib","Gefitinib","Gemcitabine",
        "Lapatinib", "MK-2206","Paclitaxel",
        "Pictilisib","Vincristine","Vorinostat")


data.info <- list(
	gCSI = c("PSet_gCSI2019.rds","gCSI"),
	GDSC2 = c("GDSC2.rds","GDSC2"),
	GDSC1 = c("PSet_GDSC2020.rds","GDSC1"),
	Broad = c("PSet_CTRPv2.rds","CTRPv2")
	)


for(ps.name in names(data.info)){
    fname <- data.info[[ps.name]][1]
    shorthand <- data.info[[ps.name]][2]
    fname <- paste0("../rawdata/psets/", fname)

    ps <- readRDS(fname)
    ps <- updateObject(ps)
    
    
    # print(names(treatmentResponse(ps)))
    
    tr <- treatmentResponse(ps)
    metadata <- tr$info %>% select(treatmentid) %>% filter(treatmentid %in% drugs)
    
    # print(head(tr$info))
    # print(head(tr$profiles))
    # print(head(tr$n))
    # print(head(tr$raw))
    # stop()

    dose.data <- merge(metadata, tr$raw[,,1],by=0)
    write.csv(dose.data,paste0("../results/",shorthand,'_dose_ranges.csv'))
}






