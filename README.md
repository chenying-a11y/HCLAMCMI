# HCLAMCMI
We proposed a model, named HCLAMCMI, to predict CMIs. The entire procedures contained three stages. In the first stage, illustrated in Module (A), three raw feature types were extracted for circRNAs and miRNAs, including two linear feature types, derived from the adjacency matrix and similarity matrices, and one non-linear feature types, obtained from a heterogeneous network. In the second stage, illustrated in Module (B) and detailed in Module (D) (E) (F), each row feature type of circRNAs or miRNAs was improved by hypergraph convolutional network, contrastive learning, and channel attention mechanism. Then, all features of circRNAs or miRNAs were individually concatenated as their complete embeddings. In the last stage, illustrated in Module (C), a two-layer full connected neural network was applied to the embeddings of circRNAs or miRNAs to generate the final representations. Then, the inner product was used to score each circRNA-miRNA pair, generating the final recommendation matrix. 

<img width="1871" height="1323" alt="Figure 1_01" src="https://github.com/user-attachments/assets/a475b567-7b2c-485d-852f-6cc693c92512" />

# Requirements
```
python==3.10.15
torch== 2.0.0
pandas==2.2.3
numpy==1.26.4
scipy==1.14.1
```
# Quick start
Run main.py to Run HCLAMCMI

