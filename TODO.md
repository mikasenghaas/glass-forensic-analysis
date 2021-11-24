# TODO 

## EDA
all eda on the entire dataset (merge train and test before)

- [ ] Numerical Summaries for all features, average, five-num-summary, variance)
- [ ] Make description of all features (quantitative vs. qualitative, small description of
      all features to have an intution on their effect in classiying the glass)
- [ ] Boxplots for all features 
- [ ] Pairplot (scatters for all pairs of features)
- [ ] Violinplot that show class distribution for all features (p violinplots, (1, k) subplots)

- [ ] Maybe make generate_markdown() function to summarise EDA for generic dataset 


## Preprocessing 
- [ ] feature selection through PCA 
- [ ] scaling features (maybe using different scaling techniques)
- [ ] generate new features (through kernels in svm) or manually through PolynomialFeatures()

## Implementation of Models

### Decision Tree
- [ ] Add more stop criterions (rn: max_depth, total purity)
      -> missing: minium leaf nodes, minimum num of datapoints per leaf, max leaves
- [ ] 

### Neural Network
- [ ] implement autograd
- [ ] write generic layer object
- [ ] write generic nn object

### Evaluate Correctness
- [ ] DT:plot decision boundaries for generic 2d-feature classification task using classification 
      results from own implementation and sklearn implementation
- [ ] NN: choose more difficult classification task and assess whether we can reach similar performance
      with our custom implementation 
 
## Training and Evaluation Pipeline
- [ ] make function that incoroporates the entire ml-training pipeline + gridsearch cv for hyper
      parameter tuning (utils)

```
def train_and_evaluate(model : mlmodel, params : dict, score: 'macro_recall'):
    # make pipeline
    # return best score, best model, best params
```

- [ ] report final performance from returned best model on test set
- [ ] build and export final model as trained from entire dataset using best test-performance with best
      selected features




