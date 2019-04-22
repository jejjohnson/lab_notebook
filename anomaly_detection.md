# Anomaly Detection

---
## RBIG for Anomaly Detection in Earth Science DataCube <a name="esdcrbig"></a>

* GitLab Repo - [Mutenroshi](https://mutenroshi.uv.es/gitlab/emmanuel/2019_rbig_ad)

This project will be fairly simply but also very applied: I am looking for anomalous events within the ESDC using the RBIG density estimation algorithm. I would like to compare it to some of the standard algorithms for multivariate anomaly detection but unfortunately there are not that many. So I will have to rely on some of the analysis done by some professionals.

### Key Pre-Processing Steps

* Remove Seasonal Trends
* Grab Appropriate Years

---
---
# Literature Review


#### AD + Theory

* A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data (2016)
* Multivariate anomaly detection for Earth observations: a comparison of algorithms and feature extraction techniques - Flach et al. (2017)
* A Comparative Evaluation of Outlier Detection Algorithms: Experiments and Analysis - Domingues et. al. (2019)

#### Algorithms

* PyOD: A Python Toolbox for Scalable Outlier Detection, Zhao et al. (2018)
* [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/outlier_detection.html)
* [Rotation-based Iterative Gaussianization (RBIG)](https://www.uv.es/vista/vistavalencia/RBIG.htm)


**Key Software**
* [List](https://github.com/rob-med/awesome-TS-anomaly-detection)
