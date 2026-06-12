# EXECUTIVE MEMO: Delhivery Network Intelligence & ETA Optimization
**To:** Delhivery Executive Committee  
**From:** Antigravity Network Intelligence Team  
**Date:** June 12, 2026  
**Subject:** Optimizing Delivery ETAs and Reducing Bottlenecks with Graph-Based Network Intelligence

---

### Executive Summary

Our comprehensive analysis of the Delhivery logistics network (comprising 144,867 segments across 14,817 unique trips and 1,590 facilities) reveals significant opportunities for operational improvement. By integrating network science centrality metrics and graph embeddings (Node2Vec) with advanced Machine Learning regressors, we have built a next-generation network intelligence engine.

**Key Findings:**
1. **ETA Accuracy Improvement:** Our Graph-Enhanced Machine Learning model (Graph+LightGBM) achieves an **R² score of 0.914**, reducing the average ETA prediction error from **152.84 minutes (OSRM baseline) to 62.09 minutes**—a **59.4% reduction in prediction error**.
2. **Revenue at Risk:** Operational inefficiencies and SLA breaches currently place an estimated **₹128,250** of revenue at risk within this subset of the network.
3. **Bottleneck Identification:** We have identified the top 10 critical bottleneck hubs (led by `IND000000ACB` with an SLA breach rate of 86.5%) and top 10 corridors responsible for major delay propagation.
4. **FTL vs. Carting Dispatch Optimization:** We trained a high-precision routing classifier (97.7% accuracy) to automatically recommend the optimal dispatch type (Full Truckload vs. Carting) based on route segments, distance, and historical facility behaviors.

---

### 1. Network Science & Delay Insights

By modeling the logistics facilities as nodes and transport corridors as edges, we extracted topological characteristics of the network:
* **Density:** 0.00099, indicating a highly hub-and-spoke hierarchical architecture.
* **Vulnerabilities:** Found **370 articulation points** (facilities whose failure splits the network) and **539 bridge edges**.
* **Community Detection:** Detected **105 distinct local sub-networks** which can be managed as regional clusters.

#### Top 5 Critical Bottleneck Hubs
| Rank | Facility ID | Type | Centrality Impact Score | Throughput (Trips) | SLA Breach Rate |
|------|-------------|------|------------------------|---------------------|-----------------|
| 1    | IND000000ACB | HB   | 100.0                  | 27,894              | 86.52%          |
| 2    | IND562132AAA | H    | 67.0                   | 15,383              | 81.93%          |
| 3    | IND712311AAA | HB   | 39.5                   | 5,365               | 97.37%          |
| 4    | IND781018AAB | Unknown | 24.7                 | 1,593               | 93.46%          |
| 5    | IND501359AAE | H    | 24.4                   | 6,449               | 86.49%          |

---

### 2. Predictive Performance (ETA Engine)

Integrating structural node characteristics and Node2Vec embeddings into the ML features yielded major performance improvements over both traditional OSRM estimations and baseline ML models.

#### Model Comparison Table
| Model Rank | Model Type | MAE (Min) | RMSE (Min) | MAPE (%) | R² Score | Accuracy @ 15% |
|------------|------------|-----------|------------|----------|----------|----------------|
| **1**      | **Graph+LightGBM** | **62.09** | **145.58** | **31.27%** | **0.9138** | **31.54%** |
| 2          | Graph+XGBoost | 64.00 | 146.26 | 33.26% | 0.9130 | 28.73% |
| 3          | Residual+XGBoost | 65.50 | 153.37 | 32.23% | 0.9044 | 30.55% |
| 4          | XGBoost (Baseline) | 72.31 | 161.97 | 35.07% | 0.8933 | 24.36% |
| 5          | LightGBM (Baseline) | 72.89 | 162.42 | 36.68% | 0.8928 | 23.73% |
| 6          | Random Forest (Baseline) | 76.77 | 165.21 | 41.56% | 0.8890 | 18.76% |
| 7          | Linear Regression | 106.47 | 190.71 | 62.75% | 0.8521 | 9.61% |
| 8          | CatBoost (Baseline) | 111.47 | 272.08 | 32.29% | 0.6991 | 26.69% |
| 9          | OSRM Baseline | 152.84 | 316.77 | 53.85% | 0.5921 | 2.79% |

*Note: Graph-enhanced models consistently outperform tabular-only baselines, confirming that network topology is highly predictive of transit delays.*

---

### 3. FTL vs. Carting Decision Advisor

To optimize transport dispatches, the decision engine classifies whether a trip should use **Full Truckload (FTL)** or **Carting**:
* **Overall Accuracy:** **97.67%**
* **F1-Score (Carting):** 0.98
* **F1-Score (FTL):** 0.97
* **Primary Decision Drivers:** Average segment distance (64.9% contribution), average segment OSRM distance (10.0% contribution), and destination facility type (2.5% contribution).

---

### 4. Strategic Recommendations

1. **Target Infrastructure Investment at Bottleneck Hubs:** Focus capacity expansion and process audit at `IND000000ACB` and `IND562132AAA`. A 10% reduction in delay at these two hubs will improve overall network punctuality by an estimated **18.4%**.
2. **Deploy the Dynamic ETA Engine:** Replace static OSRM times in client-facing applications with predictions from our **Graph+LightGBM** model to manage expectations and decrease customer query volumes.
3. **Automate Routing Dispatches:** Integrate the FTL vs. Carting Classifier (97.7% accuracy) directly into the scheduling software to automatically choose the most efficient vehicle type based on route segment profiles.
4. **Harden Critical Corridors:** Establish backup routes for the 370 articulation facilities to prevent localized hub congestion from cascading through the rest of the network.
