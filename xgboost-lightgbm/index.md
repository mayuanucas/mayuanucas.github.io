# XGBoost、LightGBM、CatBoost


XGBoost是陈天奇于2014年提出的一套并行boost算法的工具库, LightGBM是微软推出的boosting框架, CatBoost是Yandex推出的Boost工具包. 本文将对这些算法进行介绍,并在数据集上对算法进行测试.

<!-- more -->

# XGBoost

## 简介

XGBoost的全称是eXtreme Gradient Boosting，既可以用于分类也可以用于回归问题中, 它是经过优化的分布式梯度提升库，旨在高效、灵活且可移植。XGBoost是大规模并行boosting tree的工具，它是目前最快最好的开源 boosting tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量的Kaggle选手选用XGBoost进行数据挖掘比赛，是各大数据科学比赛的必杀武器；在工业界大规模数据方面，XGBoost的分布式版本有广泛的可移植性，支持在Kubernetes、Hadoop、SGE、MPI、 Dask等各个分布式环境上运行，使得它可以很好地解决工业界大规模数据的问题。
