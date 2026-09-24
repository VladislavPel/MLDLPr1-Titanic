	### To-do
- Ticket name to ticket types
- Fill nan ages with median by titles
- Making age_bands
- Making titles drop names
- Drop rooms
- One hot encoding for sex, embarking, titles

### Notes 
- women and childrens first
- 1 class survives
- cherbourg sr high
- lonely or swarmed people dies
### Questions
- pass

### Solution

| Approach                  | CV     | CV STD | LB          | Date       |
| ------------------------- | ------ | ------ | ----------- | ---------- |
| baseline                  |        |        | 0.76555     | 23.09.2026 |
| Logreg1                   | 0,8317 | 0,0105 | 0.77751     | 24.09.2026 |
| Lasso1                    | 0,8350 | 0,0092 | 0.77511     | 24.09.2026 |
| Ridge1                    | 0.8193 | 0.0130 | 0.78229     | 24.09.2026 |
| elasticnet(n=1000(error)) | 0.7979 | 0.0174 | -           | 24.09.2026 |
| elascticnet(exp5)         | 0.8058 | 0.0161 | 0.76555(??) | 24.09.2026 |
| elasticnet(exp6)          | 0.8272 | 0.0127 | 0.77511     | 24/09.2026 |
| Logreg2(e7)               | 0.8272 | 0.0127 | 0.77511     | 24.09.2026 |
| Lasso2(e8)                | 0.8249 | 0.0131 | 0.77511     | 24.09.2026 |
| knn(n=5)(e9)              | 0,798  | 0,0216 | 0,70574     | 24.09.2026 |
| dt(d=5)(e10)              | 0.8182 | 0.0204 | 0,76076     | 24.09.2026 |


### Annotations

| №   | Annotation                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | после 6 эксперимента был добавлен стандартный скейлер для Fare, придется прогонять линейные модели заново, после 8 убрал |
| 2   | knn очень плох, скорее всего дело в маленьком датасете и разбросе Fare                                                   |
| 3   | dt это сын переобучения                                                                                                  |

### Final Ensemble

| Model | CV  | Public LB | Private LB |
| ----- | --- | --------- | ---------- |
|       |     |           |            |

### Processed ideas
#### Good
- pass
#### Neutral
- pass
#### Bad
- pass
#### Some experimental ideas I didn't implemented
- pass