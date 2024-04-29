## Semi-Supervised Leanring :

For supervised leanring, we use labeled data
For unsupervised learning, we use non labeled data

In semi-supervised learning, we have both, the majority of data is not labeled while the remaining minority is labeled, because labeling data can be tedious and expensive

- Pseudo labeling : we use the labeled data in a supervised learning model, we pass to it the unlabeled data ( comme si test data) and label them
--> cons : The already labeles data amount is so small compared to the prior unlabeled data

- Self labeling : adding a confidence threshold when assinging labels to the unlabeled data

- Co-training : We split the whole labeled data ( according to featues) into 2 sections, we also split the unlabeled data into 2 sections and we pass them in a crossedd way to the models trained on the already labeled data


- Multi-training : looks like co-training but with multiple splits