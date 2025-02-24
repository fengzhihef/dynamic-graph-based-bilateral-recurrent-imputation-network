# Dynamic Graph-Based Bilateral Recurrent Imputation Network for Multivariate Time Series
This repository is an implement of the paper "Dynamic Graph-Based Bilateral Recurrent Imputation Network for Multivariate Time Series". This method aims to address the imputation of missing values in any multivariate time series without requiring a predefined graph structure. It flexibly learns the relationships between variables by dynamically constructing graph structures.

## Information of the paper
``` latex
@article{lai2025dynamic,
  title={Dynamic Graph-Based Bilateral Recurrent Imputation Network for Multivariate Time Series},
  author={Lai, Xiaochen and Zhang, Zheng and Zhang, Liyong and Lu, Wei and Li, ZhuoHan},
  journal={Neural Networks},
  pages={107298},
  year={2025},
  publisher={Elsevier}
}
```

## Framework
![image](https://github.com/fengzhihef/dynamic-graph-based-bilateral-recurrent-imputation-network/blob/main/figs/framework.jpg)


## Acknowledgement
This network is primarily implemented based on the graph recurrent imputation network (GRIN) from https://github.com/Graph-Machine-Learning-Group/grin. Our main contributions are as follows: (1) We designed an adaptive method for constructing dynamic graphs, which integrates temporal dependencies through an information fusion layer and mines localized monotonic correlations between variables using the Spearman rank correlation coefficient. These correlations are represented in segment-specific adjacency matrices, implemented in lib/nn/layers/graph_learn_module.py; (2) We utilized the dynamic graphs constructed from windowed data in a graph-based bidirectional recurrent neural network. The main model is implemented in lib/nn/models/dgbrin.py, with related network layers in lib/nn/layers; (3) We provided a convenient example that includes training, testing, and ground truth for the test data in the timeseries_datasets directory. The data is in CSV format, and the training and testing sets contain missing values. The model outputs the imputed data for the test set's missing values and calculates the imputation error based on the ground truth for the test data in the timeseries_datasets directory; (4) The model can be simply executed via main.ipynb.
