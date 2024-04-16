# [CVPR24] FedAS: Bridging Inconsistency in Personalized Fedearated Learning
Code Implementation and Informations about FedAS

> Xiyuan Yang, Wenke Huang, Mang Ye
> *CVPR, 2024*

## Abstract
Personalized Federated Learning (PFL) is primarily de- signed to provide customized models for each client to bet- ter fit the non-iid distributed client data, which is a inherent challenge in Federated Learning. However, current PFL methods suffer from inconsistencies in both intra-client and inter-client levels: 1) The intra-client inconsistency stems from the asynchronous update strategy for personalized and shared parameters. In PFL, clients update their shared pa- rameters to communicate and learn from others, while keep- ing personalized parts unchanged, leading to poor coordi- nation between these two components. 2) The Inter-client inconsistency arises from “stragglers” - inactive clients that communicate and train with the server less frequently. This results in their under-trained personalized models and impedes the collaborative training stage for other clients. In this paper, we present a novel PFL framework named FedAS, which uses Federated Parameter-Alignment and Client-Synchronization to overcome above challenges. Ini- tially, we enhance the localization of global parameters by infusing them with local insights. We make the shared parts learn from previous model, thereby increasing their local relevance and reducing the impact of parameter inconsis- tency. Furthermore, we design a robust aggregation method to mitigate the impact of stragglers by preventing the incor- poration of their under-trained knowledge into aggregated model. Experimental results on Cifar10 and Cifar100 vali- date the effectiveness of our FedAS in achieving better per- formance and robustness against data heterogeneity.


## Citation
```
@inproceedings{cvpr24_xiyuan_fedas,
    author    = {Yang, Xiyuan and Huang, Wenke and Ye, Mang},
    title     = {FedAS: Bridging Inconsistency in Personalized Fedearated Learning},
    booktitle = {CVPR},
    year      = {2024}
}
```
