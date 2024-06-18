# [CVPR24] FedAS: Bridging Inconsistency in Personalized Fedearated Learning

> Xiyuan Yang, Wenke Huang, Mang Ye
> *CVPR, 2024*
> [Paper Link](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html)
Code Implementation and Informations about FedAS

## Abstract
Personalized Federated Learning (PFL) is primarily designed to provide customized models for each client to better fit the non-iid distributed client data, which is a inherent challenge in Federated Learning. However, current PFL methods suffer from inconsistencies in both intra-client and inter-client levels: 1) The intra-client inconsistency stems from the asynchronous update strategy for personalized and shared parameters. In PFL, clients update their shared parameters to communicate and learn from others, while keeping personalized parts unchanged, leading to poor coordination between these two components. 2) The Inter-client inconsistency arises from “stragglers” - inactive clients that communicate and train with the server less frequently. This results in their undetrained personalized models and impedes the collaborative training stage for other clients. In this paper, we present a novel PFL framework named FedAS, which uses Federated Parameter-Alignment and Client-Synchronization to overcome above challenges. Initially, we enhance the localization of global parameters by infusing them with local insights. We make the shared parts learn from previous model, thereby increasing their local relevance and reducing the impact of parameter inconsistency. Furthermore, we design a robust aggregation method to mitigate the impact of stragglers by preventing the incorporation of their under-trained knowledge into aggregated model. Experimental results on Cifar10 and Cifar100 validate the effectiveness of our FedAS in achieving better performance and robustness against data heterogeneity.

## Citation
```
@inproceedings{cvpr24_xiyuan_fedas,
    author    = {Yang, Xiyuan and Huang, Wenke and Ye, Mang},
    title     = {FedAS: Bridging Inconsistency in Personalized Fedearated Learning},
    booktitle = {CVPR},
    year      = {2024}
}
```
