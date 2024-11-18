from surprise import Dataset, KNNWithMeans, Reader, SVD
from surprise.accuracy import mae
from surprise.model_selection import train_test_split, KFold, GridSearchCV
from surprise.prediction_algorithms.algo_base import AlgoBase 
from pprint import pprint
from collections import defaultdict
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

def load_csv(filename='data.csv'):
    # import csv file in python
    csv_file = pd.read_csv(filename, delimiter=';')

    # conversion of the table in a numpy matrix with only ratings
    # method .to_numpy() does not copy the header row
    # np.delete deletes the first column
    temp = np.delete(csv_file.to_numpy(), np.s_[0], axis=1)

    # Matrix is converted in User-Movies from Movies-User
    # Later, the matrix is flatten to a vector of values.
    # For each user, all the movie ratings are reported
    # user1item1, user1item2, user1item3,...
    ratings = temp.T.flatten()

    # Vectors users and movies are the corresponding columns in the dataframne.
    # As the ratings are ordered user1[allratings], user2[allratings], ...
    # the user and movies vectors follow the same logic
    i = 0
    j = 0
    users = []
    movies = []
    users.clear()
    movies.clear()
    while i < 50:
        while j < 20:
            movies.append(j)
            users.append(i)
            j += 1
        j = 0
        i += 1

    movies = np.array(movies)
    users = np.array(users)

    # The user, movies, and rating numpy vectors are converted in a rating dictionary
    # and later, in a Pandas dataframe

    ratings_dict = {'userID': users,
                    'itemID': movies,
                    'rating': ratings}

    df = pd.DataFrame(ratings_dict)

    # The dataframes are converted into a dataset suitable for Surprise

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Return precision and recall at n metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of relevant and recommended items in top n
        n_rel_and_rec = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec / n

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls

def run_knn(use_builtin: bool, test_size: float, neighbors: int, n: int):
    if use_builtin:
        data = Dataset.load_builtin('ml-100k')
    else:
        data = load_csv()

    # Dataset splitting in trainset and testset for 25% sparsity
    trainset25, testset25 = train_test_split(data, test_size=test_size,
                                            random_state=22)
    
    sim_options_KNN = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }

    # number of neighbors
    k = neighbors

    # prepare user-based KNN for predicting ratings from trainset25
    algo = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
    fit_algo(algo, trainset25, testset25, n)
    result_dict = fit_algo(algo, trainset25, testset25, n)
    result_dict["Algo"] = "KNN"
    result_dict["Dataset"] = "ml-100k" if use_builtin else "custom"
    result_dict["Neighbors"] = neighbors
    result_dict["Sparsity"] = test_size
    return result_dict

def run_svd(use_builtin: bool, test_size: float, n: int):
    if use_builtin:
        data = Dataset.load_builtin('ml-100k')
    else:
        data = load_csv()

    # Dataset splitting in trainset and testset for 25% sparsity
    trainset25, testset25 = train_test_split(data, test_size=test_size,
                                            random_state=22)
    
    # Use SVD algorithm
    algo = SVD()
    result_dict = fit_algo(algo, trainset25, testset25, n)
    result_dict["Algo"] = "SVD"
    result_dict["Dataset"] = "ml-100k" if use_builtin else "custom"
    result_dict["Sparsity"] = test_size
    return result_dict

def fit_algo(algo: AlgoBase, trainset: Dataset, testset: Dataset, n: int):
    algo.fit(trainset)

    # estimate the ratings for all the pairs (user, item) in testset25
    predictions25KNN = algo.test(testset)

    # pprint(predictions25KNN)

    # the first user has uid=0 and first item iid=0
    # for (uid, iid, real, est, _) in predictions25KNN:
        # if uid == 0:
            # print(f'{uid} {iid} {real} {est}')

    mae_value = mae(predictions25KNN, verbose=False)

    precisions, recalls = precision_recall_at_n(predictions25KNN, n=n, threshold=4)

    # Precision and recall can then be averaged over all users
    pre = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    f1 = 2*pre*recall/(pre+recall)
    # print(f"\nTOP {n} RESULTS\n")
    # print("Precision:", pre)
    # print("Recall:", recall)
    # print("F1:", f1)

    return {
        "TopN": n,
        "MAE": mae_value,
        "Precision": pre,
        "Recall": recall,
        "F1": f1
    } 

def question_1(df):
    filtered_df = df[(df['Dataset'] == 'ml-100k') & (df['Algo'] == 'KNN')]

    # Group by Sparsity and Neighbors, and calculate the mean MAE for each combination
    grouped_df = filtered_df.groupby(["Dataset", "Algo", "Sparsity", "Neighbors"])["MAE"].mean().reset_index()
    grouped_df.to_csv("output_q1.csv", index=False)

    # Create a dictionary to store optimal K values for each sparsity level
    optimal_k_dict = {}

    # Loop over each unique sparsity level
    for sparsity in grouped_df["Sparsity"].unique():
        # Filter the DataFrame for the current sparsity level
        sparse_df = grouped_df[grouped_df["Sparsity"] == sparsity]
        
        # Find the optimal K by selecting the row with the smallest MAE for each sparsity
        optimal_k = sparse_df.loc[sparse_df["MAE"].idxmin(), "Neighbors"]
        
        # Store the optimal K value in the dictionary
        optimal_k_dict[sparsity] = (optimal_k, sparse_df["MAE"].min())

    # Display the dictionary of optimal K values for each sparsity
    print("\n\n1. Optimal K values for each sparsity level:")
    for sparsity, entry in optimal_k_dict.items():
        optimal_k, min_mae = entry
        print(f"Sparsity: {sparsity}, Optimal K: {optimal_k}, Min. MAE: {min_mae}")

def question_2(df):
    filtered_df = df[(df['Dataset'] == 'ml-100k')]

    # Group by Sparsity and Neighbors, and calculate the mean MAE for each combination
    grouped_df = filtered_df.groupby(["Dataset", "Algo", "Sparsity"])["MAE"].min().reset_index()
    grouped_df.to_csv("output_q2.csv", index=False)

    # Display the dictionary of optimal K values for each sparsity
    print("\n\n2. MAE performances for KNN and SVD on built-in dataset:")
    print(grouped_df)

def question_3(df):
    filtered_df = df[(df['Dataset'] == 'ml-100k')]

    # Group by Sparsity and Neighbors, and calculate the mean MAE for each combination
    grouped_df = filtered_df.groupby(["Dataset", "Algo", "Sparsity", "TopN"]).agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean'
    }).reset_index()
    grouped_df.to_csv("output_q3.csv", index=False)

    # Display the dictionary of optimal K values for each sparsity
    print("\n\n3. Metrics performances for KNN and SVD on built-in dataset (on file)")


def run():
    # Columns must be Algo, Sparsity, Neighbors, MAE, TopN, Precision, Recall, F1
    columns = ["Dataset", "Algo", "Sparsity", "Neighbors", "MAE", "TopN", "Precision", "Recall", "F1"]

    # Create an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)

    # Create a list to hold each row dictionary
    rows = []

    # ------------------------------------------------------
    # QUESTION 1

    sparsity1 = .25
    sparsity2 = .75
    k_list = range(50,91,1)
    n = 10

    for k in tqdm(k_list):
        rows.append(run_knn(True, sparsity1, k, n))
        rows.append(run_svd(True, sparsity1, n))
    for k in tqdm(k_list):
        rows.append(run_knn(True, sparsity2, k, n))
        rows.append(run_svd(True, sparsity2, n))

    # ------------------------------------------------------
    # QUESTION 2
    def generate_k_values(sparsity_list, start_k, end_k):
        # Calculate the step size based on sparsity
        sparsity_range = len(sparsity_list) - 1
        k_values = [
            start_k + (end_k - start_k) * i / sparsity_range
            for i in range(len(sparsity_list))
        ]
        return [round(k) for k in k_values]

    sparsity_list = [.25, .35, .40, .45, .50, .60, .65, .70, .75]
    k_list = generate_k_values(sparsity_list, start_k=75, end_k=61) # 75 BEST K FOR .25, 61 BEST K FOR .75
    n = 10

    for sparsity, k in tqdm(zip(sparsity_list, k_list)):
        rows.append(run_knn(True, sparsity, k, n))
        rows.append(run_svd(True, sparsity, n))

    # ------------------------------------------------------
    # QUESTION 3
    sparsity = sparsity_list[0]
    k = k_list[0]
    n_list = range(10,101,5)

    for n in tqdm(n_list):
        rows.append(run_knn(True, sparsity, k, n))
        rows.append(run_svd(True, sparsity, n))

    sparsity = sparsity_list[-1]
    k = k_list[-1]
    for n in tqdm(n_list):
        rows.append(run_knn(True, sparsity, k, n))
        rows.append(run_svd(True, sparsity, n))

    # Convert the list of dictionaries to a DataFrame
    df = pd.json_normalize(rows)
    # Export DataFrame to a CSV file
    df.to_csv("output.csv", index=False)
    return df

if __name__ == '__main__':
    df = run()
    # df = pd.read_csv("output.csv", delimiter=',')
    question_1(df)
    question_2(df)
    question_3(df)