const tfNames = [
    "8988T|DNase|None",
    "AoSMC|DNase|None",
    "Chorion|DNase|None",
    "CLL|DNase|None",
    "Fibrobl|DNase|None",
    "FibroP|DNase|None",
    "Gliobla|DNase|None",
    "GM12891|DNase|None",
    "GM12892|DNase|None",
    "GM18507|DNase|None",
    "GM19238|DNase|None",
    "GM19239|DNase|None",
    "GM19240|DNase|None",
    "H9ES|DNase|None",
    "HeLa-S3|DNase|IFNa4h",
    "Hepatocytes|DNase|None",
    "HPDE6-E6E7|DNase|None",
    "HSMM_emb|DNase|None",
    "HTR8svn|DNase|None",
    "Huh-7.5|DNase|None",
    "Huh-7|DNase|None",
    "iPS|DNase|None",
    "Ishikawa|DNase|Estradiol_100nM_1hr",
    "Ishikawa|DNase|4OHTAM_20nM_72hr",
    "LNCaP|DNase|androgen",
    "MCF-7|DNase|Hypoxia_LacAcid",
    "Medullo|DNase|None",
    "Melano|DNase|None",
    "Myometr|DNase|None",
    "Osteobl|DNase|None",
    "PanIsletD|DNase|None",
    "PanIslets|DNase|None",
    "pHTE|DNase|None",
    "ProgFib|DNase|None",
    "RWPE1|DNase|None",
    "Stellate|DNase|None",
    "T-47D|DNase|None",
    "Adult_CD4_Th0|DNase|None",
    "Urothelia|DNase|None",
    "Urothelia|DNase|UT189",
    "AG04449|DNase|None",
    "AG04450|DNase|None",
    "AG09309|DNase|None",
    "AG09319|DNase|None",
    "AG10803|DNase|None",
    "AoAF|DNase|None",
    "BE2_C|DNase|None",
    "BJ|DNase|None",
    "Caco-2|DNase|None",
    "CD20+|DNase|None",
    "CD34+_Mobilized|DNase|None",
    "CMK|DNase|None",
    "A549|DNase|None",
    "GM12878|DNase|None",
    "H1-hESC|DNase|None",
    "HeLa-S3|DNase|None",
    "HepG2|DNase|None",
    "HMEC|DNase|None",
    "HSMMtube|DNase|None",
    "HSMM|DNase|None",
    "HUVEC|DNase|None",
    "K562|DNase|None",
    "LNCaP|DNase|None",
    "MCF-7|DNase|None",
    "NHEK|DNase|None",
    "Th1|DNase|None",
    "GM06990|DNase|None",
    "GM12864|DNase|None",
    "GM12865|DNase|None",
    "H7-hESC|DNase|None",
    "HAc|DNase|None",
    "HAEpiC|DNase|None",
    "HA-h|DNase|None",
    "HA-sp|DNase|None",
    "HBMEC|DNase|None",
    "HCFaa|DNase|None",
    "HCF|DNase|None",
    "HCM|DNase|None",
    "HConF|DNase|None",
    "HCPEpiC|DNase|None",
    "HCT-116|DNase|None",
    "HEEpiC|DNase|None",
    "HFF-Myc|DNase|None",
    "HFF|DNase|None",
    "HGF|DNase|None",
    "HIPEpiC|DNase|None",
    "HL-60|DNase|None",
    "HMF|DNase|None",
    "HMVEC-dAd|DNase|None",
    "HMVEC-dBl-Ad|DNase|None",
    "HMVEC-dBl-Neo|DNase|None",
    "HMVEC-dLy-Ad|DNase|None",
    "HMVEC-dLy-Neo|DNase|None",
    "HMVEC-dNeo|DNase|None",
    "HMVEC-LBl|DNase|None",
    "HMVEC-LLy|DNase|None",
    "HNPCEpiC|DNase|None",
    "HPAEC|DNase|None",
    "HPAF|DNase|None",
    "HPdLF|DNase|None",
    "HPF|DNase|None",
    "HRCEpiC|DNase|None",
    "HRE|DNase|None",
    "HRGEC|DNase|None",
    "HRPEpiC|DNase|None",
    "HVMF|DNase|None",
    "Jurkat|DNase|None",
    "Monocytes-CD14+_RO01746|DNase|None",
    "NB4|DNase|None",
    "NH-A|DNase|None",
    "NHDF-Ad|DNase|None",
    "NHDF-neo|DNase|None",
    "NHLF|DNase|None",
    "NT2-D1|DNase|None",
    "PANC-1|DNase|None",
    "PrEC|DNase|None",
    "RPTEC|DNase|None",
    "SAEC|DNase|None",
    "SKMC|DNase|None",
    "SK-N-MC|DNase|None",
    "SK-N-SH_RA|DNase|None",
    "Th2|DNase|None",
    "WERI-Rb-1|DNase|None",
    "WI-38|DNase|4OHTAM_20nM_72hr",
    "WI-38|DNase|None",
    "Dnd41|CTCF|None",
    "Dnd41|EZH2|None",
    "GM12878|CTCF|None",
    "GM12878|EZH2|None",
    "H1-hESC|CHD1|None",
    "H1-hESC|CTCF|None",
    "H1-hESC|EZH2|None",
    "H1-hESC|JARID1A|None",
    "H1-hESC|RBBP5|None",
    "HeLa-S3|CTCF|None",
    "HeLa-S3|EZH2|None",
    "HeLa-S3|Pol2(b)|None",
    "HepG2|CTCF|None",
    "HepG2|EZH2|None",
    "HMEC|CTCF|None",
    "HMEC|EZH2|None",
    "HSMM|CTCF|None",
    "HSMM|EZH2|None",
    "HSMMtube|CTCF|None",
    "HSMMtube|EZH2|None",
    "HUVEC|CTCF|None",
    "HUVEC|EZH2|None",
    "HUVEC|Pol2(b)|None",
    "K562|CHD1|None",
    "K562|CTCF|None",
    "K562|EZH2|None",
    "K562|HDAC1|None",
    "K562|HDAC2|None",
    "K562|HDAC6|None",
    "K562|p300|None",
    "K562|PHF8|None",
    "K562|PLU1|None",
    "K562|Pol2(b)|None",
    "K562|RBBP5|None",
    "K562|SAP30|None",
    "NH-A|CTCF|None",
    "NH-A|EZH2|None",
    "NHDF-Ad|CTCF|None",
    "NHDF-Ad|EZH2|None",
    "NHEK|CTCF|None",
    "NHEK|EZH2|None",
    "NHEK|Pol2(b)|None",
    "NHLF|CTCF|None",
    "NHLF|EZH2|None",
    "Osteobl|CTCF|None",
    "A549|ATF3|EtOH_0.02pct",
    "A549|BCL3|EtOH_0.02pct",
    "A549|CREB1|DEX_100nM",
    "A549|CTCF|DEX_100nM",
    "A549|CTCF|EtOH_0.02pct",
    "A549|ELF1|EtOH_0.02pct",
    "A549|ETS1|EtOH_0.02pct",
    "A549|FOSL2|EtOH_0.02pct",
    "A549|FOXA1|DEX_100nM",
    "A549|GABP|EtOH_0.02pct",
    "A549|GR|DEX_500pM",
    "A549|GR|DEX_50nM",
    "A549|GR|DEX_5nM",
    "A549|GR|DEX_100nM",
    "A549|NRSF|EtOH_0.02pct",
    "A549|p300|EtOH_0.02pct",
    "A549|Pol2|DEX_100nM",
    "A549|Pol2|EtOH_0.02pct",
    "A549|Sin3Ak-20|EtOH_0.02pct",
    "A549|SIX5|EtOH_0.02pct",
    "A549|TAF1|EtOH_0.02pct",
    "A549|TCF12|EtOH_0.02pct",
    "A549|USF-1|DEX_100nM",
    "A549|USF-1|EtOH_0.02pct",
    "A549|USF-1|EtOH_0.02pct",
    "A549|YY1|EtOH_0.02pct",
    "A549|ZBTB33|EtOH_0.02pct",
    "ECC-1|CTCF|DMSO_0.02pct",
    "ECC-1|ERalpha|BPA_100nM",
    "ECC-1|ERalpha|Estradiol_10nM",
    "ECC-1|ERalpha|Genistein_100nM",
    "ECC-1|FOXA1|DMSO_0.02pct",
    "ECC-1|GR|DEX_100nM",
    "ECC-1|Pol2|DMSO_0.02pct",
    "GM12878|ATF2|None",
    "GM12878|ATF3|None",
    "GM12878|BATF|None",
    "GM12878|BCL11A|None",
    "GM12878|BCL3|None",
    "GM12878|BCLAF1|None",
    "GM12878|CEBPB|None",
    "GM12878|EBF1|None",
    "GM12878|Egr-1|None",
    "GM12878|ELF1|None",
    "GM12878|ETS1|None",
    "GM12878|FOXM1|None",
    "GM12878|GABP|None",
    "GM12878|IRF4|None",
    "GM12878|MEF2A|None",
    "GM12878|MEF2C|None",
    "GM12878|MTA3|None",
    "GM12878|NFATC1|None",
    "GM12878|NFIC|None",
    "GM12878|NRSF|None",
    "GM12878|p300|None",
    "GM12878|PAX5-C20|None",
    "GM12878|PAX5-N19|None",
    "GM12878|Pbx3|None",
    "GM12878|PML|None",
    "GM12878|Pol2-4H8|None",
    "GM12878|Pol2|None",
    "GM12878|POU2F2|None",
    "GM12878|PU.1|None",
    "GM12878|Rad21|None",
    "GM12878|RUNX3|None",
    "GM12878|RXRA|None",
    "GM12878|SIX5|None",
    "GM12878|SP1|None",
    "GM12878|SRF|None",
    "GM12878|STAT5A|None",
    "GM12878|TAF1|None",
    "GM12878|TCF12|None",
    "GM12878|TCF3|None",
    "GM12878|USF-1|None",
    "GM12878|YY1|None",
    "GM12878|ZBTB33|None",
    "GM12878|ZEB1|None",
    "GM12891|PAX5-C20|None",
    "GM12891|Pol2-4H8|None",
    "GM12891|Pol2|None",
    "GM12891|POU2F2|None",
    "GM12891|PU.1|None",
    "GM12891|TAF1|None",
    "GM12891|YY1|None",
    "GM12892|PAX5-C20|None",
    "GM12892|Pol2-4H8|None",
    "GM12892|Pol2|None",
    "GM12892|TAF1|None",
    "GM12892|YY1|None",
    "H1-hESC|ATF2|None",
    "H1-hESC|ATF3|None",
    "H1-hESC|BCL11A|None",
    "H1-hESC|CTCF|None",
    "H1-hESC|Egr-1|None",
    "H1-hESC|FOSL1|None",
    "H1-hESC|GABP|None",
    "H1-hESC|HDAC2|None",
    "H1-hESC|JunD|None",
    "H1-hESC|NANOG|None",
    "H1-hESC|NRSF|None",
    "H1-hESC|p300|None",
    "H1-hESC|Pol2-4H8|None",
    "H1-hESC|Pol2|None",
    "H1-hESC|POU5F1|None",
    "H1-hESC|Rad21|None",
    "H1-hESC|RXRA|None",
    "H1-hESC|Sin3Ak-20|None",
    "H1-hESC|SIX5|None",
    "H1-hESC|SP1|None",
    "H1-hESC|SP2|None",
    "H1-hESC|SP4|None",
    "H1-hESC|SRF|None",
    "H1-hESC|TAF1|None",
    "H1-hESC|TAF7|None",
    "H1-hESC|TCF12|None",
    "H1-hESC|TEAD4|None",
    "H1-hESC|USF-1|None",
    "H1-hESC|YY1|None",
    "HCT-116|Pol2-4H8|None",
    "HCT-116|YY1|None",
    "HCT-116|ZBTB33|None",
    "HeLa-S3|GABP|None",
    "HeLa-S3|NRSF|None",
    "HeLa-S3|Pol2|None",
    "HeLa-S3|TAF1|None",
    "HepG2|ATF3|None",
    "HepG2|BHLHE40|None",
    "HepG2|CEBPB|None",
    "HepG2|CEBPD|None",
    "HepG2|CTCF|None",
    "HepG2|ELF1|None",
    "HepG2|FOSL2|None",
    "HepG2|FOXA1|None",
    "HepG2|FOXA1|None",
    "HepG2|FOXA2|None",
    "HepG2|GABP|None",
    "HepG2|HDAC2|None",
    "HepG2|HNF4A|None",
    "HepG2|HNF4G|None",
    "HepG2|JunD|None",
    "HepG2|MBD4|None",
    "HepG2|MYBL2|None",
    "HepG2|NFIC|None",
    "HepG2|NRSF|None",
    "HepG2|NRSF|None",
    "HepG2|p300|None",
    "HepG2|Pol2-4H8|None",
    "HepG2|Pol2|None",
    "HepG2|Rad21|None",
    "HepG2|RXRA|None",
    "HepG2|Sin3Ak-20|None",
    "HepG2|SP1|None",
    "HepG2|SP2|None",
    "HepG2|SRF|None",
    "HepG2|TAF1|None",
    "HepG2|TCF12|None",
    "HepG2|TEAD4|None",
    "HepG2|USF-1|None",
    "HepG2|YY1|None",
    "HepG2|ZBTB33|None",
    "HepG2|ZBTB7A|None",
    "HUVEC|Pol2-4H8|None",
    "HUVEC|Pol2|None",
    "K562|ATF3|None",
    "K562|BCL3|None",
    "K562|BCLAF1|None",
    "K562|CBX3|None",
    "K562|CEBPB|None",
    "K562|CTCF|None",
    "K562|CTCFL|None",
    "K562|E2F6|None",
    "K562|Egr-1|None",
    "K562|ELF1|None",
    "K562|ETS1|None",
    "K562|FOSL1|None",
    "K562|GABP|None",
    "K562|GATA2|None",
    "K562|HDAC2|None",
    "K562|Max|None",
    "K562|MEF2A|None",
    "K562|NR2F2|None",
    "K562|NRSF|None",
    "K562|PML|None",
    "K562|Pol2-4H8|None",
    "K562|Pol2|None",
    "K562|PU.1|None",
    "K562|Rad21|None",
    "K562|Sin3Ak-20|None",
    "K562|SIX5|None",
    "K562|SP1|None",
    "K562|SP2|None",
    "K562|SRF|None",
    "K562|STAT5A|None",
    "K562|TAF1|None",
    "K562|TAF7|None",
    "K562|TEAD4|None",
    "K562|THAP1|None",
    "K562|TRIM28|None",
    "K562|USF-1|None",
    "K562|YY1|None",
    "K562|YY1|None",
    "K562|ZBTB33|None",
    "K562|ZBTB7A|None",
    "PANC-1|NRSF|None",
    "PANC-1|Pol2-4H8|None",
    "PANC-1|Sin3Ak-20|None",
    "PFSK-1|FOXP2|None",
    "PFSK-1|NRSF|None",
    "PFSK-1|Sin3Ak-20|None",
    "PFSK-1|TAF1|None",
    "SK-N-MC|FOXP2|None",
    "SK-N-MC|Pol2-4H8|None",
    "SK-N-SH|NRSF|None",
    "SK-N-SH|NRSF|None",
    "SK-N-SH|Pol2-4H8|None",
    "SK-N-SH_RA|CTCF|None",
    "SK-N-SH_RA|p300|None",
    "SK-N-SH_RA|Rad21|None",
    "SK-N-SH_RA|USF1|None",
    "SK-N-SH_RA|YY1|None",
    "SK-N-SH|Sin3Ak-20|None",
    "SK-N-SH|TAF1|None",
    "T-47D|CTCF|DMSO_0.02pct",
    "T-47D|ERalpha|BPA_100nM",
    "T-47D|ERalpha|Genistein_100nM",
    "T-47D|ERalpha|Estradiol_10nM",
    "T-47D|FOXA1|DMSO_0.02pct",
    "T-47D|GATA3|DMSO_0.02pct",
    "T-47D|p300|DMSO_0.02pct",
    "U87|NRSF|None",
    "U87|Pol2-4H8|None",
    "A549|BHLHE40|None",
    "A549|CEBPB|None",
    "A549|Max|None",
    "A549|Pol2(phosphoS2)|None",
    "A549|Rad21|None",
    "GM08714|ZNF274|None",
    "GM10847|NFKB|TNFa",
    "GM10847|Pol2|None",
    "GM12878|BHLHE40|None",
    "GM12878|BRCA1|None",
    "GM12878|c-Fos|None",
    "GM12878|CHD1|None",
    "GM12878|CHD2|None",
    "GM12878|COREST|None",
    "GM12878|CTCF|None",
    "GM12878|E2F4|None",
    "GM12878|EBF1|None",
    "GM12878|ELK1|None",
    "GM12878|IKZF1|None",
    "GM12878|JunD|None",
    "GM12878|Max|None",
    "GM12878|MAZ|None",
    "GM12878|Mxi1|None",
    "GM12878|NF-E2|None",
    "GM12878|NFKB|TNFa",
    "GM12878|NF-YA|None",
    "GM12878|NF-YB|None",
    "GM12878|Nrf1|None",
    "GM12878|p300|None",
    "GM12878|p300|None",
    "GM12878|Pol2|None",
    "GM12878|Pol2(phosphoS2)|None",
    "GM12878|Pol2|None",
    "GM12878|Pol3|None",
    "GM12878|Rad21|None",
    "GM12878|RFX5|None",
    "GM12878|SIN3A|None",
    "GM12878|SMC3|None",
    "GM12878|STAT1|None",
    "GM12878|STAT3|None",
    "GM12878|TBLR1|None",
    "GM12878|TBP|None",
    "GM12878|TR4|None",
    "GM12878|USF2|None",
    "GM12878|WHIP|None",
    "GM12878|YY1|None",
    "GM12878|Znf143|None",
    "GM12878|ZNF274|None",
    "GM12878|ZZZ3|None",
    "GM12891|NFKB|TNFa",
    "GM12891|Pol2|None",
    "GM12892|NFKB|TNFa",
    "GM12892|Pol2|None",
    "GM15510|NFKB|TNFa",
    "GM15510|Pol2|None",
    "GM18505|NFKB|TNFa",
    "GM18505|Pol2|None",
    "GM18526|NFKB|TNFa",
    "GM18526|Pol2|None",
    "GM18951|NFKB|TNFa",
    "GM18951|Pol2|None",
    "GM19099|NFKB|TNFa",
    "GM19099|Pol2|None",
    "GM19193|NFKB|TNFa",
    "GM19193|Pol2|None",
    "H1-hESC|Bach1|None",
    "H1-hESC|BRCA1|None",
    "H1-hESC|CEBPB|None",
    "H1-hESC|CHD1|None",
    "H1-hESC|CHD2|None",
    "H1-hESC|c-Jun|None",
    "H1-hESC|c-Myc|None",
    "H1-hESC|CtBP2|None",
    "H1-hESC|GTF2F1|None",
    "H1-hESC|JunD|None",
    "H1-hESC|MafK|None",
    "H1-hESC|Max|None",
    "H1-hESC|Mxi1|None",
    "H1-hESC|Nrf1|None",
    "H1-hESC|Rad21|None",
    "H1-hESC|RFX5|None",
    "H1-hESC|SIN3A|None",
    "H1-hESC|SUZ12|None",
    "H1-hESC|TBP|None",
    "H1-hESC|USF2|None",
    "H1-hESC|Znf143|None",
    "HCT-116|Pol2|None",
    "HCT-116|TCF7L2|None",
    "HEK293|ELK4|None",
    "HEK293|KAP1|None",
    "HEK293|Pol2|None",
    "HEK293|TCF7L2|None",
    "HEK293-T-REx|ZNF263|None",
    "HeLa-S3|AP-2alpha|None",
    "HeLa-S3|AP-2gamma|None",
    "HeLa-S3|BAF155|None",
    "HeLa-S3|BAF170|None",
    "HeLa-S3|BDP1|None",
    "HeLa-S3|BRCA1|None",
    "HeLa-S3|BRF1|None",
    "HeLa-S3|BRF2|None",
    "HeLa-S3|Brg1|None",
    "HeLa-S3|CEBPB|None",
    "HeLa-S3|c-Fos|None",
    "HeLa-S3|CHD2|None",
    "HeLa-S3|c-Jun|None",
    "HeLa-S3|c-Myc|None",
    "HeLa-S3|COREST|None",
    "HeLa-S3|E2F1|None",
    "HeLa-S3|E2F4|None",
    "HeLa-S3|E2F6|None",
    "HeLa-S3|ELK1|None",
    "HeLa-S3|ELK4|None",
    "HeLa-S3|GTF2F1|None",
    "HeLa-S3|HA-E2F1|None",
    "HeLa-S3|Ini1|None",
    "HeLa-S3|IRF3|None",
    "HeLa-S3|JunD|None",
    "HeLa-S3|MafK|None",
    "HeLa-S3|Max|None",
    "HeLa-S3|MAZ|None",
    "HeLa-S3|Mxi1|None",
    "HeLa-S3|NF-YA|None",
    "HeLa-S3|NF-YB|None",
    "HeLa-S3|Nrf1|None",
    "HeLa-S3|p300|None",
    "HeLa-S3|Pol2(phosphoS2)|None",
    "HeLa-S3|Pol2|None",
    "HeLa-S3|PRDM1|None",
    "HeLa-S3|Rad21|None",
    "HeLa-S3|RFX5|None",
    "HeLa-S3|RPC155|None",
    "HeLa-S3|SMC3|None",
    "HeLa-S3|SPT20|None",
    "HeLa-S3|STAT1|IFNg30",
    "HeLa-S3|STAT3|None",
    "HeLa-S3|TBP|None",
    "HeLa-S3|TCF7L2|None",
    "HeLa-S3|TCF7L2|None",
    "HeLa-S3|TFIIIC-110|None",
    "HeLa-S3|TR4|None",
    "HeLa-S3|USF2|None",
    "HeLa-S3|ZKSCAN1|None",
    "HeLa-S3|Znf143|None",
    "HeLa-S3|ZNF274|None",
    "HeLa-S3|ZZZ3|None",
    "HepG2|ARID3A|None",
    "HepG2|BHLHE40|None",
    "HepG2|BRCA1|None",
    "HepG2|CEBPB|forskolin",
    "HepG2|CEBPB|None",
    "HepG2|CHD2|None",
    "HepG2|c-Jun|None",
    "HepG2|COREST|None",
    "HepG2|ERRA|forskolin",
    "HepG2|GRp20|forskolin",
    "HepG2|HNF4A|forskolin",
    "HepG2|HSF1|forskolin",
    "HepG2|IRF3|None",
    "HepG2|JunD|None",
    "HepG2|MafF|None",
    "HepG2|MafK|None",
    "HepG2|MafK|None",
    "HepG2|Max|None",
    "HepG2|MAZ|None",
    "HepG2|Mxi1|None",
    "HepG2|Nrf1|None",
    "HepG2|p300|None",
    "HepG2|PGC1A|forskolin",
    "HepG2|Pol2|forskolin",
    "HepG2|Pol2|None",
    "HepG2|Pol2(phosphoS2)|None",
    "HepG2|Rad21|None",
    "HepG2|RFX5|None",
    "HepG2|SMC3|None",
    "NA|NA|NA",
    "HepG2|TBP|None",
    "HepG2|TCF7L2|None",
    "HepG2|TR4|None",
    "HepG2|USF2|None",
    "HepG2|ZNF274|None",
    "HUVEC|c-Fos|None",
    "HUVEC|c-Jun|None",
    "HUVEC|GATA-2|None",
    "HUVEC|Max|None",
    "HUVEC|Pol2|None",
    "IMR90|CEBPB|None",
    "IMR90|CTCF|None",
    "IMR90|MafK|None",
    "IMR90|Pol2|None",
    "IMR90|Rad21|None",
    "K562|ARID3A|None",
    "K562|ATF1|None",
    "K562|ATF3|None",
    "K562|Bach1|None",
    "K562|BDP1|None",
    "K562|BHLHE40|None",
    "K562|BRF1|None",
    "K562|BRF2|None",
    "K562|Brg1|None",
    "K562|CCNT2|None",
    "K562|CEBPB|None",
    "K562|c-Fos|None",
    "K562|CHD2|None",
    "K562|c-Jun|IFNa30",
    "K562|c-Jun|IFNa6h",
    "K562|c-Jun|IFNg30",
    "K562|c-Jun|IFNg6h",
    "K562|c-Jun|None",
    "K562|c-Myc|IFNa30",
    "K562|c-Myc|IFNa6h",
    "K562|c-Myc|IFNg30",
    "K562|c-Myc|IFNg6h",
    "K562|c-Myc|None",
    "K562|c-Myc|None",
    "K562|COREST|None",
    "K562|COREST|None",
    "K562|CTCF|None",
    "K562|E2F4|None",
    "K562|E2F6|None",
    "K562|ELK1|None",
    "K562|GATA-1|None",
    "K562|GATA-2|None",
    "K562|GTF2B|None",
    "K562|GTF2F1|None",
    "K562|HMGN3|None",
    "K562|Ini1|None",
    "K562|IRF1|IFNa30",
    "K562|IRF1|IFNa6h",
    "K562|IRF1|IFNg30",
    "K562|IRF1|IFNg6h",
    "K562|JunD|None",
    "K562|KAP1|None",
    "K562|MafF|None",
    "K562|MafK|None",
    "K562|Max|None",
    "K562|MAZ|None",
    "K562|Mxi1|None",
    "K562|NELFe|None",
    "K562|NF-E2|None",
    "K562|NF-YA|None",
    "K562|NF-YB|None",
    "K562|Nrf1|None",
    "K562|p300|None",
    "K562|Pol2|IFNa30",
    "K562|Pol2|IFNa6h",
    "K562|Pol2|IFNg30",
    "K562|Pol2|IFNg6h",
    "K562|Pol2|None",
    "K562|Pol2(phosphoS2)|None",
    "K562|Pol2(phosphoS2)|None",
    "K562|Pol2|None",
    "K562|Pol3|None",
    "K562|Rad21|None",
    "K562|RFX5|None",
    "K562|RPC155|None",
    "K562|SETDB1|MNaseD",
    "K562|SETDB1|None",
    "K562|SIRT6|None",
    "K562|SMC3|None",
    "K562|STAT1|IFNa30",
    "K562|STAT1|IFNa6h",
    "K562|STAT1|IFNg30",
    "K562|STAT1|IFNg6h",
    "K562|STAT2|IFNa30",
    "K562|STAT2|IFNa6h",
    "K562|TAL1|None",
    "K562|TBLR1|None",
    "K562|TBLR1|None",
    "K562|TBP|None",
    "K562|TFIIIC-110|None",
    "K562|TR4|None",
    "K562|UBF|None",
    "K562|UBTF|None",
    "K562|USF2|None",
    "K562|YY1|None",
    "K562|Znf143|None",
    "K562|ZNF263|None",
    "K562|ZNF274|None",
    "K562|ZNF274|None",
    "MCF10A-Er-Src|c-Fos|EtOH_0.01pct",
    "MCF10A-Er-Src|c-Fos|4OHTAM_1uM_12hr",
    "MCF10A-Er-Src|c-Fos|4OHTAM_1uM_4hr",
    "MCF10A-Er-Src|c-Fos|4OHTAM_1uM_36hr",
    "MCF10A-Er-Src|c-Myc|EtOH_0.01pct",
    "MCF10A-Er-Src|c-Myc|4OHTAM_1uM_4hr",
    "MCF10A-Er-Src|E2F4|4OHTAM_1uM_36hr",
    "MCF10A-Er-Src|Pol2|EtOH_0.01pct",
    "MCF10A-Er-Src|Pol2|4OHTAM_1uM_36hr",
    "MCF10A-Er-Src|STAT3|EtOH_0.01pct_4hr",
    "MCF10A-Er-Src|STAT3|EtOH_0.01pct_12hr",
    "MCF10A-Er-Src|STAT3|EtOH_0.01pct",
    "MCF10A-Er-Src|STAT3|4OHTAM_1uM_12hr",
    "MCF10A-Er-Src|STAT3|4OHTAM_1uM_36hr",
    "MCF-7|GATA3|None",
    "MCF-7|GATA3|None",
    "MCF-7|HA-E2F1|None",
    "MCF-7|TCF7L2|None",
    "MCF-7|ZNF217|None",
    "NB4|c-Myc|None",
    "NB4|Max|None",
    "NB4|Pol2|None",
    "NT2-D1|SUZ12|None",
    "NT2-D1|YY1|None",
    "NT2-D1|ZNF274|None",
    "PANC-1|TCF7L2|None",
    "PBDEFetal|GATA-1|None",
    "PBDE|GATA-1|None",
    "PBDE|Pol2|None",
    "Raji|Pol2|None",
    "SH-SY5Y|GATA-2|None",
    "SH-SY5Y|GATA3|None",
    "U2OS|KAP1|None",
    "U2OS|SETDB1|None",
    "K562|eGFP-FOS|None",
    "K562|eGFP-GATA2|None",
    "K562|eGFP-HDAC8|None",
    "K562|eGFP-JunB|None",
    "K562|eGFP-JunD|None",
    "A549|CTCF|None",
    "A549|Pol2|None",
    "Fibrobl|CTCF|None",
    "Gliobla|CTCF|None",
    "Gliobla|Pol2|None",
    "GM12878|c-Myc|None",
    "GM12878|CTCF|None",
    "GM12878|Pol2|None",
    "GM12891|CTCF|None",
    "GM12892|CTCF|None",
    "GM19238|CTCF|None",
    "GM19239|CTCF|None",
    "GM19240|CTCF|None",
    "H1-hESC|c-Myc|None",
    "H1-hESC|CTCF|None",
    "H1-hESC|Pol2|None",
    "HeLa-S3|c-Myc|None",
    "HeLa-S3|CTCF|None",
    "HeLa-S3|Pol2|None",
    "HepG2|c-Myc|None",
    "HepG2|CTCF|None",
    "HepG2|Pol2|None",
    "HUVEC|c-Myc|None",
    "HUVEC|CTCF|None",
    "HUVEC|Pol2|None",
    "K562|c-Myc|None",
    "K562|CTCF|None",
    "K562|Pol2|None",
    "MCF-7|c-Myc|estrogen",
    "MCF-7|c-Myc|serum_stimulated_media",
    "MCF-7|c-Myc|serum_starved_media",
    "MCF-7|c-Myc|vehicle",
    "MCF-7|CTCF|estrogen",
    "MCF-7|CTCF|serum_stimulated_media",
    "MCF-7|CTCF|serum_starved_media",
    "MCF-7|CTCF|None",
    "MCF-7|CTCF|vehicle",
    "MCF-7|Pol2|serum_stimulated_media",
    "MCF-7|Pol2|serum_starved_media",
    "MCF-7|Pol2|None",
    "NHEK|CTCF|None",
    "ProgFib|CTCF|None",
    "ProgFib|Pol2|None",
    "A549|CTCF|None",
    "AG04449|CTCF|None",
    "AG04450|CTCF|None",
    "AG09309|CTCF|None",
    "AG09319|CTCF|None",
    "AG10803|CTCF|None",
    "AoAF|CTCF|None",
    "BE2_C|CTCF|None",
    "BJ|CTCF|None",
    "Caco-2|CTCF|None",
    "GM06990|CTCF|None",
    "GM12801|CTCF|None",
    "GM12864|CTCF|None",
    "GM12865|CTCF|None",
    "GM12872|CTCF|None",
    "GM12873|CTCF|None",
    "GM12874|CTCF|None",
    "GM12875|CTCF|None",
    "GM12878|CTCF|None",
    "HAc|CTCF|None",
    "HA-sp|CTCF|None",
    "HBMEC|CTCF|None",
    "HCFaa|CTCF|None",
    "HCM|CTCF|None",
    "HCPEpiC|CTCF|None",
    "HCT-116|CTCF|None",
    "HEEpiC|CTCF|None",
    "HEK293|CTCF|None",
    "HeLa-S3|CTCF|None",
    "HepG2|CTCF|None",
    "HFF|CTCF|None",
    "HFF-Myc|CTCF|None",
    "HL-60|CTCF|None",
    "HMEC|CTCF|None",
    "HMF|CTCF|None",
    "HPAF|CTCF|None",
    "HPF|CTCF|None",
    "HRE|CTCF|None",
    "HRPEpiC|CTCF|None",
    "HUVEC|CTCF|None",
    "HVMF|CTCF|None",
    "K562|CTCF|None",
    "MCF-7|CTCF|None",
    "NB4|CTCF|None",
    "NHDF-neo|CTCF|None",
    "NHEK|CTCF|None",
    "NHLF|CTCF|None",
    "RPTEC|CTCF|None",
    "SAEC|CTCF|None",
    "SK-N-SH_RA|CTCF|None",
    "WERI-Rb-1|CTCF|None",
    "WI-38|CTCF|None",
    "H1-hESC|H2AK5ac|None",
    "H1-hESC|H2AZ|None",
    "H1-hESC|H2BK120ac|None",
    "H1-hESC|H2BK12ac|None",
    "H1-hESC|H2BK15ac|None",
    "H1-hESC|H2BK20ac|None",
    "H1-hESC|H2BK5ac|None",
    "H1-hESC|H3K14ac|None",
    "H1-hESC|H3K18ac|None",
    "H1-hESC|H3K23ac|None",
    "H1-hESC|H3K23me2|None",
    "H1-hESC|H3K27ac|None",
    "H1-hESC|H3K27me3|None",
    "H1-hESC|H3K36me3|None",
    "H1-hESC|H3K4ac|None",
    "H1-hESC|H3K4me1|None",
    "H1-hESC|H3K4me2|None",
    "H1-hESC|H3K4me3|None",
    "H1-hESC|H3K56ac|None",
    "H1-hESC|H3K79me1|None",
    "H1-hESC|H3K79me2|None",
    "H1-hESC|H3K9ac|None",
    "H1-hESC|H3K9me3|None",
    "H1-hESC|H4K20me1|None",
    "H1-hESC|H4K5ac|None",
    "H1-hESC|H4K8ac|None",
    "H1-hESC|H4K91ac|None",
    "K562|H2AZ|None",
    "K562|H3K27ac|None",
    "K562|H3K27me3|None",
    "K562|H3K36me3|None",
    "K562|H3K4me1|None",
    "K562|H3K4me2|None",
    "K562|H3K4me3|None",
    "K562|H3K79me2|None",
    "K562|H3K9ac|None",
    "K562|H3K9me1|None",
    "K562|H3K9me3|None",
    "K562|H4K20me1|None",
    "Monocytes-CD14+RO01746 |H2AZ|None",
    "Monocytes-CD14+RO01746 |H3K27ac|None",
    "Monocytes-CD14+RO01746 |H3K27me3|None",
    "Monocytes-CD14+RO01746 |H3K36me3|None",
    "Monocytes-CD14+RO01746 |H3K4me1|None",
    "Monocytes-CD14+RO01746 |H3K4me2|None",
    "Monocytes-CD14+RO01746 |H3K4me3|None",
    "Monocytes-CD14+RO01746 |H3K79me2|None",
    "Monocytes-CD14+RO01746 |H3K9ac|None",
    "Monocytes-CD14+RO01746 |H3K9me3|None",
    "Monocytes-CD14+RO01746 |H4K20me1|None",
    "NH-A|H2AZ|None",
    "NH-A|H3K27ac|None",
    "NH-A|H3K27me3|None",
    "NH-A|H3K36me3|None",
    "NH-A|H3K4me1|None",
    "NH-A|H3K4me2|None",
    "NH-A|H3K4me3|None",
    "NH-A|H3K79me2|None",
    "NH-A|H3K9ac|None",
    "NH-A|H3K9me3|None",
    "NH-A|H4K20me1|None",
    "NHDF-Ad|H2AZ|None",
    "NHDF-Ad|H3K27ac|None",
    "NHDF-Ad|H3K27me3|None",
    "NHDF-Ad|H3K36me3|None",
    "NHDF-Ad|H3K4me1|None",
    "NHDF-Ad|H3K4me2|None",
    "NHDF-Ad|H3K4me3|None",
    "NHDF-Ad|H3K79me2|None",
    "NHDF-Ad|H3K9ac|None",
    "NHDF-Ad|H3K9me3|None",
    "NHDF-Ad|H4K20me1|None",
    "NHEK|H2AZ|None",
    "NHEK|H3K27ac|None",
    "NHEK|H3K27me3|None",
    "NHEK|H3K36me3|None",
    "NHEK|H3K4me1|None",
    "NHEK|H3K4me2|None",
    "NHEK|H3K4me3|None",
    "NHEK|H3K79me2|None",
    "NHEK|H3K9ac|None",
    "NHEK|H3K9me1|None",
    "NHEK|H3K9me3|None",
    "NHEK|H4K20me1|None",
    "NHLF|H2AZ|None",
    "NHLF|H3K27ac|None",
    "NHLF|H3K27me3|None",
    "NHLF|H3K36me3|None",
    "NHLF|H3K4me1|None",
    "NHLF|H3K4me2|None",
    "NHLF|H3K4me3|None",
    "NHLF|H3K79me2|None",
    "NHLF|H3K9ac|None",
    "NHLF|H3K9me3|None",
    "NHLF|H4K20me1|None",
    "Osteoblasts|H2AZ|None",
    "Osteoblasts|H3K27ac|None",
    "Osteoblasts|H3K27me3|None",
    "Osteoblasts|H3K36me3|None",
    "Osteoblasts|H3K4me1|None",
    "Osteoblasts|H3K4me2|None",
    "Osteoblasts|H3K4me3|None",
    "Osteoblasts|H3K79me2|None",
    "Osteoblasts|H3K9me3|None"
]

export default tfNames;