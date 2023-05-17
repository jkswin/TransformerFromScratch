# Transformer From Scratch

This started as a _Transformer from Scratch_ project, but turned into an _Everything from Scratch_ project. 

It is now my default repo for making scratch-implementations for learning purposes/when I'm bored. 

![scratch](md_resources/scratch.png)

Content so far:

- [Toy Pokemon & YuGiOh Data Loader](transformer/toy_data/dataset_wrapper.py)
- [Forward Layer Implementation](transformer/neural.py) & [Test Neural Net](transformer/test_neural.py) with:
    - Layer Normalization
    - [He and Xavier Initialisation](transformer/utils/activation_functions.py)
    - Outputs for Simple, Binary Classification and Multiclass Classification problems
- [Simple Logistic Regression](transfomer/regression.py) to [compare to NN for Pokemon Classification](transformer/test_regression.py) (bonus linear regression content as a stepping stone to logistic)