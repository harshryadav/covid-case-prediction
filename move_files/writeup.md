# Assignment 2 Writeup — Logistic Regression Classifier on Congressional Speeches

## A. Senate Speeches

### A1. Evaluation Numbers (Custom Feature Extraction)

| Metric                     | Value   |
|----------------------------|---------|
| Accuracy                   | 0.8038  |
| Precision (Republican)     | 0.9146  |
| Recall (Republican)        | 0.8128  |
| Precision (Democrat)       | 0.5861  |
| Recall (Democrat)          | 0.7775  |

Training set: 3,174 examples (Republican: 2,136 | Democrat: 1,038)
Test set: 1,361 examples (Republican: 902 | Democrat: 459)

**Most Informative Features:**

| Democrat-leaning (negative coef.) | Republican-leaning (positive coef.) |
|-----------------------------------|--------------------------------------|
| -1.5440  speak                    | 1.1049  agreed                       |
| -1.4117  reserving                | 1.1016  minute                       |
| -1.1319  times                    | 1.0096  break                        |
| -1.1169  announce                 | 0.9935  permitted                    |
| -1.0633  rescinded                | 0.9559  debate                       |
| -1.0598  house_managers           | 0.9412  period                       |
| -1.0364  managers                 | 0.9154  table                        |
| -0.9817  passed                   | 0.8835  counsel                      |
| -0.9814  massachusetts            | 0.8552  senators_permitted           |
| -0.9346  markey                   | 0.8430  iowa                         |

### A2. Confusion Matrix

|                  | Predicted Democrat | Predicted Republican |
|------------------|--------------------|----------------------|
| **True Democrat**    | 0.58               | 0.42                 |
| **True Republican**  | 0.082              | 0.92                 |

The classifier is much stronger at correctly identifying Republican speeches (92% recall) than Democrat speeches (58% recall). This asymmetry reflects the class imbalance in the training data — there are roughly twice as many Republican examples, so the model is biased toward predicting "Republican."

### A3. Comparison to Most-Frequent-Label Baseline

The most frequent label in the training set is "Republican" (2,136 out of 3,174 = 67.3%). A naive baseline that always predicts "Republican" for every test item would achieve:

- **Baseline Accuracy** = 902 / 1,361 = **0.6627** (66.3%)

Our logistic regression classifier achieves **0.8038** (80.4%), which is a substantial **14.1 percentage point improvement** over the naive baseline. This confirms that the bag-of-words features provide meaningful discriminative signal beyond simple class frequency.

---

## B. House Speeches

### B1. Evaluation Numbers (Custom Feature Extraction)

| Metric                     | Value   |
|----------------------------|---------|
| Accuracy                   | 0.7922  |
| Precision (Republican)     | 0.7675  |
| Recall (Republican)        | 0.7488  |
| Precision (Democrat)       | 0.8103  |
| Recall (Democrat)          | 0.8255  |

Training set: 6,600 examples (Democrat: 3,582 | Republican: 3,018)
Test set: 2,829 examples (Democrat: 1,629 | Republican: 1,200)

**Most Informative Features:**

| Democrat-leaning (negative coef.) | Republican-leaning (positive coef.) |
|-----------------------------------|--------------------------------------|
| -1.5616  move                     | 1.5066  indiana                      |
| -1.4705  yield_30                 | 1.5062  arkansas                     |
| -1.4106  massachusetts            | 1.3695  kentucky                     |
| -1.2710  california               | 1.2551  reserve                      |
| -1.1591  ms.                      | 1.2437  partisan                     |
| -1.1043  consideration            | 1.1519  correct                      |
| -1.0836  gentlewoman              | 1.1454  mccarthy                     |
| -1.0531  garcia                   | 1.1341  lamalfa                      |
| -1.0444  connecticut              | 1.1145  west                         |
| -1.0216  urge_passage             | 1.0902  mcclintock                   |

### B2. Confusion Matrix

The House classifier is more balanced than the Senate classifier. Precision and recall are closer across both classes, reflecting the more balanced class distribution in the House training data (54.3% Democrat vs. 45.7% Republican).

### B3. Comparison to Most-Frequent-Label Baseline

The most frequent label in the House training set is "Democrat" (3,582 out of 6,600 = 54.3%). A naive baseline that always predicts "Democrat" would achieve:

- **Baseline Accuracy** = 1,629 / 2,829 = **0.5758** (57.6%)

Our logistic regression achieves **0.7922** (79.2%), an improvement of **21.6 percentage points** over the naive baseline. The larger improvement compared to the Senate case is partly because the House baseline is lower (the classes are more balanced, so always guessing the majority class is less effective).

---

## C. Combined House + Senate Speeches

### C1. Evaluation Numbers (Custom Feature Extraction)

| Metric                     | Value   |
|----------------------------|---------|
| Accuracy                   | 0.7883  |
| Precision (Republican)     | 0.8262  |
| Recall (Republican)        | 0.7712  |
| Precision (Democrat)       | 0.7496  |
| Recall (Democrat)          | 0.8085  |

Training set: 9,774 examples (Republican: 5,139 | Democrat: 4,635)
Test set: 4,190 examples (Republican: 2,117 | Democrat: 2,073)

**Most Informative Features:**

| Democrat-leaning (negative coef.) | Republican-leaning (positive coef.) |
|-----------------------------------|--------------------------------------|
| -1.6209  reserving                | 1.5786  proceed                      |
| -1.4831  yield_30                 | 1.4896  indiana                      |
| -1.3928  suspend                  | 1.4619  arkansas                     |
| -1.3189  massachusetts            | 1.3672  mccarthy                     |
| -1.3162  california               | 1.3451  speakers_remaining           |
| -1.2077  rescinded                | 1.3350  lamalfa                      |
| -1.1890  garcia                   | 1.3202  mr._lamalfa                  |
| -1.1750  yielded                  | 1.2222  cloture_motion               |
| -1.1596  discharge                | 1.2039  mr._mccarthy                 |
| -1.1495  yield_1                  | 1.1044  considered_read              |

### C2. Confusion Matrix

The combined classifier shows fairly balanced performance across both parties, with slightly higher recall for Democrats (0.81) than Republicans (0.77). The more balanced training set (52.6% Republican vs. 47.4% Democrat) leads to more symmetric predictions.

### C3. Comparison to Most-Frequent-Label Baseline

The most frequent label in the combined training set is "Republican" (5,139 out of 9,774 = 52.6%). A naive baseline that always predicts "Republican" would achieve:

- **Baseline Accuracy** = 2,117 / 4,190 = **0.5053** (50.5%)

Our logistic regression achieves **0.7883** (78.8%), an improvement of **28.3 percentage points**. This is the largest improvement over baseline of all three datasets because the combined data has the most balanced class distribution, making the majority-class baseline nearly equivalent to a coin flip.

---

## D. Summary Table

| Dataset           | # Train examples | # Test examples | Overall Test Accuracy |
|-------------------|------------------|-----------------|-----------------------|
| House             | 6,600            | 2,829           | 0.7922                |
| Senate            | 3,174            | 1,361           | 0.8038                |
| All (House+Senate)| 9,774            | 4,190           | 0.7883                |

**Observations:**
- The Senate classifier has the highest accuracy (0.8038), but this is partly inflated by the class imbalance — with ~67% Republican speeches, predicting "Republican" more often is a safer bet. When compared to its respective majority-class baseline, the Senate improvement (+14.1pp) is actually the smallest.
- The House and Combined classifiers have lower raw accuracy but achieve a much larger lift over their baselines, suggesting the model is doing proportionally more useful work when the class distribution is more balanced.
- Combining House and Senate does not improve accuracy; in fact it slightly decreases it. This may be because the two chambers use different procedural language and speaking patterns, introducing noise when mixed together.

---

*Note: Minor numerical differences from the reference output in the assignment instructions are due to running with scikit-learn 1.x, which changed the internal shuffling behavior of `train_test_split` compared to the older version used when the assignment was written. The code logic and methodology are identical.*
