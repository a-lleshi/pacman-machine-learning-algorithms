# pacman-machine-learning-algorithms
In this repo you will find a variety of machine learning algorithms used to control and manipulate PacMan.

## Machine Learning Classifers:
Within the classifier folder you will find a file named `classifier.py` in here I include the following four commonly used machine learning classifiers:
1. Naive Bayes
2. KNN (K-Nearest Neighbors) Classifier
3. Decision Tree Classifier
4. SVM (Support Vector Machine) Classifier

These classifers can be run with the following command:
## Commands to run 
```python
python3 pacman.py -p ClassifierAgent
```


## Machine Leaning Q-Learning Alogirhtm:
Within the q-learing folder I have developed a Q-learning algorithm. This can be found inside the file named `mlLearningAgents.py`.

The q-learning perform better than the classifers. This can be run using the following command:
```python

python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

```
