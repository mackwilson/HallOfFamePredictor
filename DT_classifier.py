import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, model_selection, metrics, feature_selection
from sklearn.tree import export_graphviz
import csv
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the last field is a class label 'Y' or 'N' for inducted or not
	- the remaining fields are real-valued numbers 
    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for inducted, and 0 for not inducted
    @param dataset_path: full path of the dataset CSV file
    @return
	X,y
    '''
    
    data = []
    with open (dataset_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    
    # Store the file's shape as variables to use later.
    num_examples = len(data)
    num_features = len(data[1])

    file_as_array = np.asarray(data, dtype=str)
    feature_names = file_as_array[0,1:num_features-1]
    
    # Create an array to store all file data except the class labels (last col).
    X = file_as_array[1:, 1:num_features-1] 
    for i in range(X.shape[0]):
        for j in range (X.shape[1]):
            if X[i,j] == '\\N' or X[i,j] == '':
                X[i,j] = 0
            if j==X.shape[1]-1:
                s = X[i,j]
                X[i,j]=s[:4]
    
    # Create a 1D array to store all the class labels.
    y = np.zeros_like(file_as_array[1:,1], dtype=int)
    for i in range(len(file_as_array[:,1])-1):
        # Store a binary 1 for elected, 0 for just nominated and not inducted
        y[i] = (file_as_array[i+1,num_features-1]=='Y')
    
    return X.astype(float),y.astype(float), feature_names #convert to float at the end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def print_prediction_report(clf, X_test, y_true, names, metric):
    '''
    Return a bunch of statistics and metrics reporting the performance of a 
     certain classifier model on the given training data. 
     
    @param:
        y_true: A np-array of the target class labels as integers
        y_pred: A np-array of the classifier-predicted class labels as integers
        names: A tuple of the class labels (str), corresponding to (1,0) 
               binary integer class labels
               
    @return:
        None. Print to console. 
    '''
    
    labels = (1,0)
    y_pred = clf.predict(X_test)
    
    print('\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('BEST ITERATION INFO FOR {}'.format(metric))
        
    # Confusion matrix.
    print('\nConfusion Matrix:') 
    cm = metrics.confusion_matrix(y_true, y_pred, labels)
    assert len(names)==len(cm)
    assert cm.shape == (2,2)   
    print('{:14} {:10} {:10} {:3}'.format('PREDICTED:',names[0], names[1], 'All'))
    print("ACTUAL: ")
    print('{:14} {:3} {:3} {:1} {:2} {:3} {:5}'.format(names[0], '(TP)', cm[0,0], '','(FN)', cm[0,1], sum(cm[0])))
    print('{:14} {:3} {:3} {:1} {:2} {:3} {:5}'.format(names[1], '(FP)', cm[1,0], '','(TN)', cm[1,1], sum(cm[1])))
    print('{:14} {:8} {:10} {:5}'.format('All',sum(cm[:,0]), sum(cm[:,1]), sum(sum(cm))))
    
    # Classification report.
    print("\nClassification Report:")
    print(metrics.classification_report(y_true, y_pred, labels, target_names=names))
    
    # Miscellaneous metrics.
    print("\nOverall Metrics:")
    print('{:14} {:.2f}'.format('accuracy:', metrics.accuracy_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('precision:', metrics.precision_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('recall:', metrics.recall_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('f1:', metrics.f1_score(y_true, y_pred) ))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def classify(impurity, X_train, y_train, X_test):
    clf = tree.DecisionTreeClassifier(criterion=impurity)
    clf = clf.fit(X_train,y_train)
    return clf.predict(X_test), clf
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def output_csv(accuracy, predictions):
    np.savetxt('output_files/g69_DT_{}_accuracy.csv'.format(imp), np.asarray(accuracy), 
        fmt='%i', delimiter=',', header="Dataset number, Accuracy", comments='')

    np.savetxt('output_files/g69_DT_{}_predictions.csv'.format(imp), np.asarray(predictions), 
        fmt='%i', delimiter=',', header="Iteration, Classification, Prediction", comments='')



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def select_best_features(X, y, k, names):
    # Select the best features
    best_features = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=k)
    scores = np.asarray(best_features.fit(X, y).scores_)
    X_best = best_features.transform(X)

    # Print out the features used
    scores_copy = scores.copy()
    top_features = []
    for i in range(k):
        idx = np.argmax(scores_copy)
        scores_copy[idx] = 0
        top_features.append(names[idx])

    print('\nTop {} features to use for classification: '.format(k))
    print(top_features)

    for i in range(len(names)): 
        if names[i] == "yearsActiveBatting":
            names[i] = "yBat"
        if names[i] == "yearsActiveFielding": names[i] = "yField"
        if names[i] == "yearsActivePitching": names[i]="yPit"
        if names[i] == "GBatting": names[i] = "GBat"
        if names[i] == "GPitching": names[i]="GPit"
        if names[i] == "GFielding": names[i]="GFie"
        
    ind = range(len(names))
    plt.rc('font', size=30)
    plt.bar(ind, scores, width=1)
    plt.ylabel("Scores")
    plt.xlabel("Feature")
    plt.title("Feature Selection using SelectKBest")
    plt.xticks(ind, names, rotation=90)
    plt.show()

    return X_best


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_decision_tree(clf, names, classes):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = names,class_names=classes)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('output_files/DT_{}.png'.format(imp))
    Image(graph.create_png())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    # Change these as required. 
    class_labels = ("Elected", "Nominated") #corresponding to (1,0) binary vals
    path_to_data = 'output_files/extracted_features.csv'
    test_set_ratio = 0.2
    iterations=5
    numFeatures = 10
    
    # Pre-process the dataset.
    data_full, labels, feature_names = prepare_dataset(path_to_data)
    print('The dataset is {}% nominated'.format((sum(labels)/len(labels))*100))

    data = select_best_features(data_full, labels, numFeatures, feature_names)
    

    # Split the dataset into the corresponding ratio for crossvalidation. 
    # Set random_state to a hard-coded number to ensure repeatability.
    for imp in ["gini", "entropy"]:
        metric_accuracy = []
        predictions = []
        classifications = []
        testLen = int(0.2*len(labels)*iterations)+iterations
        metric_predictions = np.zeros((testLen, 3))
        bestClf = []
        bestAcc = 0

        for i in range(iterations):
            train_data,test_data,train_labels,test_labels = model_selection.train_test_split(
                    data, labels, test_size=test_set_ratio, random_state=2*i)  
            pred_labels, clf = classify(imp, train_data, train_labels, test_data)
            acc = metrics.accuracy_score(test_labels, pred_labels)*100
            if acc > bestAcc:
                bestAcc = acc
                bestClf = clf
            metric_accuracy.append([i+1, acc])
            predictions.append(pred_labels)
            classifications.append(test_labels)

            start = i*len(pred_labels)
            end = start+len(pred_labels)
            metric_predictions[start:end, 0] = i+1
            metric_predictions[start:end, 1] = test_labels
            metric_predictions[start:end, 2] = pred_labels
        
        plot_decision_tree(bestClf, feature_names, class_labels)
        output_csv(metric_accuracy, metric_predictions)
        print_prediction_report(bestClf, test_data, test_labels, class_labels, imp)
        print("\nDecision Tree has {} nodes".format(bestClf.tree_.node_count))
  
