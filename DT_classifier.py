import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, model_selection, metrics, feature_selection
import csv


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


def print_prediction_report(y_pred, y_true, names, metric):
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
    
    print('\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('LAST ITERATION INFO FOR {}'.format(metric))
        
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
    return clf.predict(X_test)
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def output_csv(accuracy, predictions):
    np.savetxt('g69_DT_{}_accuracy.csv'.format(imp), np.asarray(accuracy), 
        fmt='%i', delimiter=',', header="Dataset number, Accuracy", comments='')

    np.savetxt('g69_DT_{}_predictions.csv'.format(imp), np.asarray(predictions), 
        fmt='%i', delimiter=',', header="Iteration, Classifictaion, Prediction", comments='')



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def select_best_features(X, y, k, names):
    # Select the best features
    best_features = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=k)
    scores = np.asarray(best_features.fit(X, y).scores_)
    X_best = best_features.transform(X)

    # Print out the features used
    top_features = []
    for i in range(k):
        idx = np.argmax(scores)
        scores[idx] = 0
        top_features.append(names[idx])

    print('\nTop {} features to use for classification: '.format(k))
    print(top_features)

    return X_best


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    # Change these as required. 
    class_labels = ("Elected", "Nominated") #corresponding to (1,0) binary vals
    path_to_data = 'input.csv'
    test_set_ratio = 0.2
    iterations=5
    numFeatures = 10
    
    # Pre-process the dataset.
    data_full, labels, feature_names = prepare_dataset(path_to_data)

    data = select_best_features(data_full, labels, numFeatures, feature_names)
    

    # Split the dataset into the corresponding ratio for crossvalidation. 
    # Set random_state to a hard-coded number to ensure repeatability.
    for imp in ["gini", "entropy"]:
        metric_accuracy = []
        predictions = []
        classifications = []
        testLen = int(0.2*len(labels)*iterations)+iterations
        metric_predictions = np.zeros((testLen, 3))

        for i in range(iterations):
            train_data,test_data,train_labels,test_labels = model_selection.train_test_split(
                    data, labels, test_size=test_set_ratio, random_state=2*i)  
            pred_labels = classify(imp, train_data, train_labels, test_data)

            metric_accuracy.append([i, metrics.accuracy_score(test_labels, pred_labels)*100])
            predictions.append(pred_labels)
            classifications.append(test_labels)

            start = i*len(pred_labels)
            end = start+len(pred_labels)
            metric_predictions[start:end, 0] = i+1
            metric_predictions[start:end, 1] = test_labels
            metric_predictions[start:end, 2] = pred_labels
        

        output_csv(metric_accuracy, metric_predictions)
        print_prediction_report(pred_labels, test_labels, class_labels, imp)

            

    # TODO: selelctKBest features 
  
