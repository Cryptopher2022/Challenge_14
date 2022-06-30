
# *Challenge 14 - README.md File*
---
These few questions were pulled out of the body of the Instructions and have been answered up front.  PNG files are provided to show the slight change to the results.  

Answer the following question: What impact resulted from increasing or decreasing the training window?

*The results were less accurate than the original short and long SMA.* 

Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

*For the iterations (shown below), they were very close but candidly not much change.  The only actual change was a decrease in accuracy.*

Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

*(Shown below - all iterations)  The original short and long SMA proved to be the most accurate.  


# Instructions
Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:

1. Establish a Baseline Performance

2. Tune the Baseline Trading Algorithm

3. Evaluate a New Machine Learning Classifier

4. Create an Evaluation Report

Establish a Baseline Performance
In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four.

Import the OHLCV dataset into a Pandas DataFrame.

Generate trading signals using short- and long-window SMA values.

Split the data into training and testing datasets.

Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

Review the classification report associated with the SVC model predictions.

Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

Write your conclusions about the performance of the baseline trading algorithm in the README.md file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.

Tune the Baseline Trading Algorithm
In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:

Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your README.md file. Answer the following question: What impact resulted from increasing or decreasing the training window?
HINT
Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your README.md file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

    *The optimal way to solve this and find the best solution would be to create a for or while loop and go through each iteration in as granular detail as possible.  Changing one of the SMA's at a time, possibly 5 at a time to hone in on the right area and then smaller increments to fine the Apex for that one variable.  Repeat with the second variable and then back and forth or create a 2 variable optimization program to find the optimal balance and solve for the highest accuracy.*

Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

What was performed was the following:

Original Round:

1. Short 4
2. Long 100

2nd Round:

1. Short 2
2. Long 100

3rd Round:

1. Short 1
2. Long 100

4th Round:

1. Short 4
2. Long 50

Additionally, the alternate method of classification I utilized was Logistic Regression.  I added that analysis to each round above and will display the findings below.  

        *There was no material change in the 4 iterations that I performed*

# Evaluate a New Machine Learning Classifier
I used Logistic Regression as an alternate Classifier, but was not able to obtain better results.  Both sets of results: The original SVM classifier (even with 4 iterations) and the LogisticRegression came up with essentially identical results.  It's a little disappointing but perhaps my eye is not keen enough to decipher the fine nuances of the differences.  I focused on the accuracy, which did not vary much in any of the iterations.  

I followed the instructions, as found below:

    In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:

    Import a new classifier, such as AdaBoost, DecisionTreeClassifier, or LogisticRegression. (For the full list of classifiers, refer to the Supervised learning page (Links to an external site.) in the scikit-learn documentation.)

    Using the original training data as the baseline model, fit another model with the new classifier.

    Backtest the new model to evaluate its performance. 

Create an Evaluation Report
As mentioned throughout this README files, the differrences were not dramatic.  I was already late and actually spent Monday - Wednesday tweaking the model and making sure it was being done correctly and applying the LogisticRegression along side.  

1. If time were not an issue:

    A. I would have built the quasi-Monte Carlo simulator to solve for the peak values for the long and the short SMA.  

    B. I would then have selected several alternate methods of classifying the data in order to optimize the findings, as buided by the backtesting.  

Shown below are the results of the 4 differing iterations:

# Iteration 1 - Original Short and Long SMA

![Original Data](images/Original_ShortSMA_LongSMA%20-%20first%20before%20all.png)

![Original Classification Report](/images/Original_shortSma)

![Original Chart]()

![Original variables with LogisticRegression]()

# Iteration 2

# Iteration 3

# Iteration 4



---

## Technologies
This program was built entirely in tandem with the prepared questions in a Jupyter Notebook using Python and the associated libraries noted above.  It was also built using Windows 10 on a Dell Laptop PC.  


---

## Installation Guide

### *This is how the libraries are imported into the program.  These import statements reside at the top of the code and are executed first.*

Written in python and utiizing the following libraries:

    import pandas as pd
    import numpy as np
    from pathlib import Path
    import hvplot.pandas
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from pandas.tseries.offsets import DateOffset
    from sklearn.metrics import classification_report 

pandas: https://pandas.pydata.org/

numpy: https://numpy.org/

Path (from pathlib): https://docs.python.org/3/library/pathlib.html

hvplot: https://hvplot.holoviz.org/user_guide/Introduction.html

matplotlib.pyplot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

sklearn: https://scikit-learn.org/

StandardScaler from sklearn.preprocessing: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

DateOffset from pandas.tseries.offsets: https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html

classification_report from sklearn.metrics: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html



## Contributors

All of the work was performed by Christopher Todd Garner

---

## License

You may use this code as you see fit as long as any copy and paste is done so with proper sourcing of materials back to this repository.                                      
