MDWGAN-GP: Data Augmentation for GeneExpression Data based on Multiple Discriminator
WGAN-GP

# Architecture

![MDWGANGP2](image\MDWGANGP2.png)

# Data sources

TCGA and GTEx dataset:https://github.com/mskcc/RNAseqDB/tree/master/data/normalized

PPI

HumanNet V3:https://staging2.inetbio.org/humannetv3/download.php

String :https://cn.string-db.org/cgi/download?sessionId=b9sL0FbgoiPo



Homo sapiens reference genome database pipeline GRCh38 GRCh38 (https://www.ncbi.nlm.nih.gov/genome/annotation_euk/Homo_sapiens/ 106/) 

Genes (ENSG), transcripts (ENST) and proteins (ENSP),  **interconversion**

# Runing Processes

1.conda create -n your_env_name python=3.8.0

2.conda activate your_env_name

3. **Note You do not need to install all packages. Select the required packages.**

  #conda install --yes --file requirements.txt

```
python main.py
```



# References

- Thanks to the following authors for their papers and codes.

  [1] VIAS R, ANDRS-TERR H, LI P, et al. Adversarial generation of gene expression data [J]. Bioinformatics, 2022, 38(3): 730-7.

  [2] TRAN N-T, TRAN V-H, NGUYEN N-B, et al. On data augmentation for gan training [J]. IEEE Transactions on Image Processing, 2021, 30: 1882-97.

  [3] WANG J, MA A, CHANG Y, et al. scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses [J]. Nature communications, 2021, 12(1): 1-11.

  
