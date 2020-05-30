import pandas as pd
import numpy as np
import csv
import datetime

Train = pd.read_csv('sample_train.csv')
Dev = pd.read_csv('sample_dev.csv')
Test = pd.read_csv('sample_test.csv')

#Calculate the entropy of a dataset.
def Entropy(target):
    result, count = np.unique(target, return_counts = True)
    Entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(result))])
    return Entropy

#Calculate the information gain of a feature.
def Infogain(data, split_feature, target="class"):
    Entropy1= Entropy(data[target])
    value, count = np.unique(data[split_feature], return_counts=True)
    Entropy2 = np.sum(
        [(count[i] / np.sum(count)) * Entropy(data.where(data[split_feature] == value[i]).dropna()[target])
         for i in range(len(value))])
    Gain = Entropy1 - Entropy2
    return Gain

#Building the tree
def Treebuilding(data, originaldata, features, target="class", parent_node=None):
    # stopping cases
    # If all target_values have the same value
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    # If the dataset is empty
    elif len(data) == 0:
        return np.unique(originaldata[target])[
            np.argmax(np.unique(originaldata[target], return_counts=True)[1])]
    # No more features
    elif len(features) == 0:
        return parent_node

    # tree growing
    else:
        parent_node_class = np.unique(data[target])[
            np.argmax(np.unique(data[target], return_counts=True)[1])]

        # feature which has biggest info gain
        Features_Gain = [ Infogain(data, feature, target) for feature in
                       features]  # Return the information gain values for the features in the dataset
        split_feature_index = np.argmax(Features_Gain)
        split_feature = features[split_feature_index]

        # Tree structure using feature of best gain
        tree = {split_feature: {}}

        # execlude feature with the best info gain
        features = [i for i in features if i != split_feature]

        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[split_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[split_feature] == value).dropna()

            #  for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = Treebuilding(sub_data, data, features, target, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[split_feature][value] = subtree
        return (tree)

def traverse(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return traverse(query, result)
            else:
                return result

def Getaccuracy(data, tree):
    # Create new query instances  removing the target feature column from the original dataset and convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")
    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])
    # Calculate the prediction accuracy
    for i in range(len(data)):
        result = traverse(queries[i], tree, 1.0)
        predicted.loc[i, "predicted"] = result
        #print(result)
    return((np.sum(predicted["predicted"] == data["rating"]) / len(data)) * 100)

def test(data, tree):
    output = open("Test predictions.txt", "w")
    for i in range(len(data)):
        result = traverse(data.iloc[i], tree,1.0)
        output.write(str(result))
        output.write("\n")
    output.close()

#Drive code
now = datetime.datetime.now()
print('start of training',now.hour-12,":",now.minute,":",now.second)
tree = Treebuilding(Train, Train, Train.columns[:-2] ,'rating')
now = datetime.datetime.now()
print('end of training',now.hour-12,":",now.minute,":",now.second)
#Accuracy of Train dataset
print('The prediction accuracy of Train dataset is: ',Getaccuracy(Train, tree), '%')
#Accuracy of Dev dataset
print('The prediction accuracy of Dev dataset is: ',Getaccuracy(Dev, tree) , '%')
#get predictions file of test dataset
test(Test, tree)
#prompt the user to enter the features of a new text sample and print the prediction for this sample.
while 1:
    print('Enter a line to predict the rating')
    line = input()
    list1 = line.split(",")
    keylist = pd.read_csv("sample_test.csv", nrows=1)

    with open('new line review.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=keylist)
        writer.writeheader()
        writer.writerow(dict(zip(keylist, list1)))

    Sample = pd.read_csv('new line review.csv')
    for i in range(len(Sample)):
        result = traverse(Sample.iloc[i], tree, 1.0)

    print(result)

