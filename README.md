# adaptive-systems-assignment-2

You must do this assignment in teams of 2 or 3. 

The assignment consists of the following tasks:

1. Given this [data set](https://grouplens.org/datasets/movielens/100k/) and the algorithm of K-NN explained in class for user-based CF:

    a) Find out the value for K that minimizes the MAE with 25% of missing ratings.

    b) Sparsity Problem: find out the value for K that minimizes the MAE with 75% of missing ratings.

2. Mitigation of sparsity problem: show how SVD (Funk variant) can provide a better MAE than user-based K-NN using the provided [data](https://moodle.upm.es/titulaciones/oficiales/mod/resource/view.php?id=267778) set.

3. Top-N recommendations: calculate the precision, recall, and F1 with different values for N (10..100) using user-based K-NN (with the best Ks)  and SVD. To do this, you must suppose that the relevant recommendations for a specific user are those rated with 4 or 5 stars in the data set. Perform the calculations for both 25% and 75% of missing ratings.

Explain why you think that the results reported in the three tasks make sense.

To do this assignment, you have to use the python library Surprise, which provides implementations for the CF algorithms addressed in the course.

You will have to submit a zip/rar file containing a document explaining the results obtained in the proposed tasks and the discussion of these results, as well as the source code. The usage of graphs to show the data obtained from the scripts will be positively assessed.