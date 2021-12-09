# TODO 

## EDA
all eda on the entire dataset (merge train and test before)

- [ ] Numerical Summaries for all features, average, five-num-summary, variance)
- [ ] Make description of all features (quantitative vs. qualitative, small description of
      all features to have an intution on their effect in classiying the glass)
- [x] Boxplots for all features 
- [x] Pairplot (scatters for all pairs of features)
- [x] Violinplot that show class distribution for all features (p violinplots, (1, k) subplots)

- [ ] Maybe make generate_markdown() function to summarise EDA for generic dataset 


## Preprocessing 
- [ ] feature selection through PCA 
- [ ] scaling features (maybe using different scaling techniques)
- [ ] generate new features (through kernels in svm) or manually through PolynomialFeatures()

## Implementation of Models

- [ ] transfer plotting function from `eduml` to this project (and use to plot decision boundaries)
- [ ] define custom errors, sa. ModelNotFittedError
- [ ] more checkers and validators to ensure data integrity and useful error messages

### Make Base Model Classifier
- [ ] have string -> int mapping defined
- [ ] commonly used attributes
- [ ] is_fitted method

### Decision Tree
- [ ] Add more stop criterions 
      - [x] max depth
      - [ ] maximum leaf nodes
      - [ ] maximum nodes 
      - [x] min samples per leaf 
- [ ] plot decision tree (or basic plain text summary)
- [x] implement predict_proba() function
- [ ] make testings of all features (including visualising) 
    - [ ] different max_depths
    - [ ] random feature selection 
- [ ] test different criterion functions

### Neural Network
- [ ] generate more test cases to check performance (correctness and speed of implementation)
- [ ] maybe: generalise model to only take hidden layers (and infer input and output layer from fitted data)
      -> here we need to make a design choice: either only add layers on call to .fit() or ask for n_features, n_classes
         during initialisation
- [ ] implement 'softmax' activation for dense layer
- [ ] change dense layer, st. it only takes number of neurons (ie. infers number of input neurons automatically)
- [ ] implement mapping to account for generic target vectors
- [ ] check correctness of other loss-functions (mse and cross-entropy currently not working)

- [ ] implement one-hot-encoding in helpers (to not be dependent on sklearn's implementation)

### Plotting
- [x] make plot_2d_decision_function() check for predict_proba function of model and 
      plot sizes according to predict proba

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
