############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Decision Trees - Random Forest

# Citation: 
# PEREIRA, V. (2018). Project: Random Forest, File: Python-DM-Classification-05-RF.py, GitHub repository: <https://github.com/Valdecy/Random-Forest>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np
import random
import matplotlib.pyplot as plt
import itertools
from random import randint
from sklearn.metrics import confusion_matrix
from copy import deepcopy

# Function: Returns True, if a Column is Numeric
def is_number(string):
    for i in range(0, len(string)):        
        if pd.isnull(string.iloc[i]) == False: 
            try:
                float(string.iloc[i])
                return True
            except ValueError:
                return False

# Function: Returns True, if a Value is Numeric
def is_number_value(value):
    if pd.isnull(value) == False:          
        try:
            float(value)
            return True
        except ValueError:
            return False

# Function: Prediction           
def prediction_dt_rf(model, Xdata):
    Xdata = Xdata.reset_index(drop=True)
    remove_redundant_rules = True
    model = model[0]
    ydata = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["Prediction"])
    for j in range(0, ydata.shape[1]):
        if ydata.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
            for i in range(0, ydata.shape[0]):          
               if ydata.iloc[i,j] == 0:
                   ydata.iloc[i,j] = "zero"
               else:
                   ydata.iloc[i,j] = "one"
    pred  = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["Prediction"])
    for i in range(0, len(model[len(model)-1])):
        label = pd.DataFrame(0, index=range(0, Xdata.shape[0]), columns=[model[len(model)-1][i]])
        pred  = pd.concat([pred, label], axis = 1)
    data  = pd.concat([ydata, Xdata], axis = 1)
    rule = []
    for j in range(0, data.shape[1]):
        if data.iloc[:,j].dtype == "bool":
            data.iloc[:,j] = data.iloc[:, j].astype(str)
    
    dt_model = deepcopy(model)

    count = 0
    end_count = data.shape[1]
    while (count < end_count-1):
        count = count + 1
        if is_number(data.iloc[:, 1]) == False:
            col_name = data.iloc[:, 1].name
            new_col  = data.iloc[:, 1].unique()
            for k in range(0, len(new_col)):
                one_hot_data = data.iloc[:, 1]
                one_hot_data = pd.DataFrame({str(col_name) + "[" + str(new_col[k]) + "]": data.iloc[:, 1]})
                for L in range (0, one_hot_data.shape[0]):
                    if one_hot_data.iloc[L, 0] == new_col[k]:
                        one_hot_data.iloc[L, 0] = " 1 "
                    else:
                        one_hot_data.iloc[L, 0] = " 0 "
                data = pd.concat([data, one_hot_data.astype(np.int32)], axis = 1)
            data.drop(col_name, axis = 1, inplace = True)
            end_count = data.shape[1]
        else:
            col_name = data.iloc[:, 1].name
            one_hot_data = data.iloc[:, 1]
            data.drop(col_name, axis = 1, inplace = True)
            data = pd.concat([data, one_hot_data], axis = 1)
    
    # Preprocessing - Binary Values
    for i in range(0, data.shape[0]):
        for j in range(1, data.shape[1]):
            if data.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
               if data.iloc[i,j] == 0:
                   data.iloc[i,j] = str(0)
               else:
                   data.iloc[i,j] = str(1)
    
    for i in range(0, len(dt_model) - 1):
        for j in range(0, len(dt_model[i])):
            dt_model[i][j] = dt_model[i][j].replace("{", "")
            dt_model[i][j] = dt_model[i][j].replace("}", "")
            dt_model[i][j] = dt_model[i][j].replace(";", "")
            dt_model[i][j] = dt_model[i][j].replace("IF ", "")
            dt_model[i][j] = dt_model[i][j].replace("AND", "")
            dt_model[i][j] = dt_model[i][j].replace("THEN", "")
            dt_model[i][j] = dt_model[i][j].replace("=", "")
            dt_model[i][j] = dt_model[i][j].replace("<", "<=")
            dt_model[i][j] = dt_model[i][j].replace(" 0 ", "<=0")
            dt_model[i][j] = dt_model[i][j].replace(" 1 ", ">0")
        
    for i in range(0, len(dt_model) - 1):
        for j in range(0, len(dt_model[i]) - 2): 
            splited_rule = [x for x in dt_model[i][j].split(" ") if x]
            rule.append(splited_rule)
    
    if remove_redundant_rules == True:
        rule = [list(x) for x in set(tuple(x) for x in rule)]
    
    for i in range(0, Xdata.shape[0]): 
        for j in range(0, len(rule)):
            rule_confirmation = len(rule[j])/2 - 1
            rule_count = 0
            for k in range(0, len(rule[j]) - 2, 2):
                if (rule[j][k] in list(data.columns.values)) == False:
                    zeros = pd.DataFrame(0, index = range(0, data.shape[0]), columns = [rule[j][k]])
                    data  = pd.concat([data, zeros], axis = 1)
                if is_number_value(data[rule[j][k]][i]) == False:
                    if (data[rule[j][k]][i] in rule[j][k+1]):
                        rule_count = rule_count + 1
                        if (rule_count == rule_confirmation):
                            pred.at[pred.index[i], rule[j][len(rule[j]) - 1]] += 1
                    else:
                        k = len(rule[j])
                elif is_number_value(data[rule[j][k]][i]) == True:
                     if rule[j][k+1].find("<=") == 0:
                         if float(data[rule[j][k]][i]) <= float(rule[j][k+1].replace("<=", "")): 
                             rule_count = rule_count + 1
                             if (rule_count == rule_confirmation):
                                 pred.at[pred.index[i], rule[j][len(rule[j]) - 1]] += 1
                         else:
                             k = len(rule[j])
                     elif rule[j][k+1].find(">") == 0:
                         if float(data[rule[j][k]][i]) > float(rule[j][k+1].replace(">", "")): 
                             rule_count = rule_count + 1
                             if (rule_count == rule_confirmation):
                                 pred.at[pred.index[i], rule[j][len(rule[j]) - 1]] += 1
                         else:
                             k = len(rule[j])
    
    for i in range(0, pred.shape[0]):
        for j in range(1, pred.shape[1]):
            if pred.iloc[i][j] == pred.max(axis=1)[i]:
                pred.at[i,"Prediction"] = pred.columns[j]
    
    pred  = pd.concat([pred, pd.DataFrame(0, index=range(0, Xdata.shape[0]), columns=["Fired Rules"])], axis = 1)
    for i in range(0, pred.shape[0]):
        for j in range(1, pred.shape[1] - 1):
            pred.at[i,"Fired Rules"] += pred.iloc[i][j]
    
    for i in range(0, pred.shape[0]):
        if pred.at[i, "Fired Rules"] == 0:
            pred.at[i, "Prediction"] = "No Rule was Fired"
    
    return pred, Xdata

# Function: Calculates oob error estimates
def oob_error_estimates(model, Xdata, ydata):
    oob_list = model[1]   
    name = ydata.name
    #classes = ydata.unique()
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    for j in range(0, ydata.shape[1]):
        if ydata.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
            for i in range(0, ydata.shape[0]):          
               if ydata.iloc[i,j] == 0:
                   ydata.iloc[i,j] = "zero"
               else:
                   ydata.iloc[i,j] = "one"
    ydata.columns = [name]
    classes = ydata.iloc[:,0].unique()
    oob_observations = pd.DataFrame({'Observations' : pd.Series(range(0,Xdata.shape[0]))})       
    oob_pred  = pd.DataFrame(index=range(0, Xdata.shape[0]), columns=["Prediction"])
    oob_pred  = pd.concat([oob_observations, ydata, oob_pred], axis = 1)
 
    for i in range(0, Xdata.shape[0]):
        sliced_model = deepcopy(model)
        for j in range(len(oob_list)-1, -1, -1):
            if i not in oob_list[j]:      
                del sliced_model[0][j]
        if len(sliced_model[0]) != 1:
            p = prediction_dt_rf(sliced_model, Xdata.iloc[[i]])
            oob_pred.at[oob_pred.index[i], "Prediction"] = p[0].iloc[0, 0]
        else:
            oob_pred.at[oob_pred.index[i], "Prediction"] = "No Rule was Fired"
    
    count = 0
    adjustment = 0
    for i in range(0, oob_pred.shape[0]):
        if oob_pred.at[i, "Prediction"] == "No Rule was Fired":
            adjustment = adjustment + 1
        if oob_pred.at[i, "Prediction"] == oob_pred.iloc[i, 1]  and oob_pred.at[i, "Prediction"] != "No Rule was Fired":
            count = count + 1
   
    if oob_pred.shape[0] - adjustment == 0:
        error = 1.
    else:
        error = 1 - (count)/(oob_pred.shape[0] - adjustment)
    
    oob_pred = oob_pred[oob_pred.Prediction.str.contains("No Rules Left") == False]
    cm = confusion_matrix(oob_pred.iloc[:,1], oob_pred.iloc[:,2], labels = classes)
    row_sum = np.sum(cm, axis = 1) 
    string = " Class Errors ==> "
    for i in range(0, len(classes)):
        for j in range(0, len(classes)):
            if i == j and row_sum[i] > 0:
                string = string + classes[i]  + " = " + '{:.2%}'.format(1 - (cm[i,j]/row_sum[i])) + "  "
    oob_error_e = ("OOB Error Estimate = " + '{:.2%}'.format(error))
    print(oob_error_e, string) 

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
   
    return oob_pred

# Function: Calculates the Gini Index  
def gini_index(target, feature = [], uniques = []):
    gini = 0
    weighted_gini = 0
    denominator_1 = feature.count()
    data = pd.concat([pd.DataFrame(target.values.reshape((target.shape[0], 1))), feature], axis = 1)
    for word in range(0, len(uniques)):
        denominator_2 = feature[(feature == uniques[word])].count() #12
        if denominator_2[0] > 0:
            for lbl in range(0, len(np.unique(target))):
                numerator_1 = data.iloc[:,0][(data.iloc[:,0] == np.unique(target)[lbl]) & (data.iloc[:,1]  == uniques[word])].count()
                if numerator_1 > 0:
                    gini = gini + (numerator_1/denominator_2)**2 
        gini = 1 - gini
        weighted_gini = weighted_gini + gini*(denominator_2/denominator_1)
        gini = 0
    return float(weighted_gini)

# Function: Binary Split on Continuous Variables 
def split_me(feature, split):
    result = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    for fill in range(0, len(feature)):
        result.iloc[fill,0] = feature.iloc[fill]
    lower = "<=" + str(split)
    upper = ">" + str(split)
    for convert in range(0, len(feature)):
        if float(feature.iloc[convert]) <= float(split):
            result.iloc[convert,0] = lower
        else:
            result.iloc[convert,0] = upper
    binary_split = []
    binary_split = [lower, upper]
    return result, binary_split

# Function: RF Algorithm
def dt_rf(Xdata, ydata, cat_missing = "none", num_missing = "none", forest_size = 5, m_attributes = 0):
    
    ################     Part 1 - Preprocessing    #############################
    forest_size = int(forest_size)
    if forest_size <= 0:
        forest_size = 1

    if is_number_value(forest_size) == False:
        forest_size = 5    
    
    if m_attributes == 0:
        m_attributes = int(Xdata.shape[1]**(1/2))
    else:
        m_attributes = m_attributes
        
    if m_attributes > Xdata.shape[1]:
        m_attributes = int(Xdata.shape[1]**(1/2))
        
    if is_number_value(m_attributes) == False:
        m_attributes = int(Xdata.shape[1]**(1/2))
        
    n_rows = Xdata.shape[0] - int(Xdata.shape[0]*(2/3))
    # Preprocessing - Creating Dataframe
    name = ydata.name        
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))   
    
    # Preprocessing - Binary Values
    for j in range(0, ydata.shape[1]):
        if ydata.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
            for i in range(0, ydata.shape[0]):          
               if ydata.iloc[i,j] == 0:
                   ydata.iloc[i,j] = "zero"
               else:
                   ydata.iloc[i,j] = "one"
    
    dataset = pd.concat([ydata, Xdata], axis = 1)  
                      
     # Preprocessing - Boolean Values
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:,j].dtype == "bool":
            dataset.iloc[:,j] = dataset.iloc[:, j].astype(str)

    # Preprocessing - Missing Values
    if cat_missing != "none":
        for j in range(1, dataset.shape[1]): 
            if is_number(dataset.iloc[:, j]) == False:
                for i in range(0, dataset.shape[0]):
                    if pd.isnull(dataset.iloc[i,j]) == True:
                        if cat_missing == "missing":
                            dataset.iloc[i,j] = "Unknow"
                        elif cat_missing == "most":
                            dataset.iloc[i,j] = dataset.iloc[:,j].value_counts().idxmax()
                        elif cat_missing == "remove":
                            dataset = dataset.drop(dataset.index[i], axis = 0)
                        elif cat_missing == "probability":
                            while pd.isnull(dataset.iloc[i,j]) == True:
                                dataset.iloc[i,j] = dataset.iloc[randint(0, dataset.shape[0] - 1), j]            
    elif num_missing != "none":
            if is_number(dataset.iloc[:, j]) == True:
                for i in range(0, dataset.shape[0]):
                    if pd.isnull(dataset.iloc[i,j]) == True:
                        if num_missing == "mean":
                            dataset.iloc[i,j] = dataset.iloc[:,j].mean()
                        elif num_missing == "median":
                            dataset.iloc[i,j] = dataset.iloc[:,j].median()
                        elif num_missing == "most":
                            dataset.iloc[i,j] = dataset.iloc[:,j].value_counts().idxmax()
                        elif cat_missing == "remove":
                            dataset = dataset.drop(dataset.index[i], axis = 0)
                        elif num_missing == "probability":
                            while pd.isnull(dataset.iloc[i,j]) == True:
                                dataset.iloc[i,j] = dataset.iloc[randint(0, dataset.shape[0] - 1), j]  

    # Preprocessing - One Hot Encode
    count = 0
    end_count = dataset.shape[1]
    while (count < end_count-1):
        count = count + 1
        if is_number(dataset.iloc[:, 1]) == False:
            col_name = dataset.iloc[:, 1].name
            new_col  = dataset.iloc[:, 1].unique()
            for k in range(0, len(new_col)):
                one_hot_data = dataset.iloc[:, 1]
                one_hot_data = pd.DataFrame({str(col_name) + "[" + str(new_col[k]) + "]": dataset.iloc[:, 1]})
                for L in range (0, one_hot_data.shape[0]):
                    if one_hot_data.iloc[L, 0] == new_col[k]:
                        one_hot_data.iloc[L, 0] = " 1 "
                    else: 
                        one_hot_data.iloc[L, 0] = " 0 "
                dataset = pd.concat([dataset, one_hot_data.astype(np.int32)], axis = 1)
            dataset.drop(col_name, axis = 1, inplace = True)
            end_count = dataset.shape[1]
        else:
            col_name = dataset.iloc[:, 1].name
            one_hot_data = dataset.iloc[:, 1]
            dataset.drop(col_name, axis = 1, inplace = True)
            dataset = pd.concat([dataset, one_hot_data], axis = 1)

    full_names = list(dataset)
    bin_names = list(dataset)
    for i in range(0, len(full_names)):
        full_names[i] = str(full_names[i]).split('[', 1)[0]
     
    # Preprocessing - Binary Values
    for i in range(0, dataset.shape[0]):
        for j in range(1, dataset.shape[1]):
            if dataset.iloc[:,j].dropna().value_counts().index.isin([0,1]).all():
               bin_names[j] = "binary"
               if dataset.iloc[i,j] == 0:
                   dataset.iloc[i,j] = str(0)
               else:
                   dataset.iloc[i,j] = str(1)
    
    original  = dataset.copy(deep = True)
    forest    = [None]*1
    oob_samples = [None]*1
    
    ################     Part 2 - RF Algorithm    #############################
    for tree in range(0, forest_size):
        dataset = original.copy()

        # Preprocessing - 2/3 of N
        drop_rows = random.sample(list(dataset.index.values), n_rows)
       
        if tree == 0:
             oob_samples[tree] = drop_rows
        else:
            oob_samples.append(drop_rows)
            
        dataset.drop(drop_rows, axis = 0, inplace = True) 
    
        # Preprocessing - Unique Words List
        unique = []
        uniqueWords = []
        for j in range(0, dataset.shape[1]): 
            for i in range(0, dataset.shape[0]):
                token = dataset.iloc[i, j]
                if not token in unique:
                    unique.append(token)
            uniqueWords.append(unique)
            unique = []  
        
        # Preprocessing - Label Matrix
        label = np.array(uniqueWords[0])
        label = label.reshape(1, len(uniqueWords[0]))
        
        ################    Part 3 - Initialization    #############################
        # RF - Initializing Variables
        i = 0
        branch = [None]*1
        branch[0] = dataset
        gini_vector = np.empty([1, branch[i].shape[1]])
        lower = " 0 "
        root_index = 0
        rule = [None]*1
        rule[0] = "IF "
        skip_update = False
        stop = 2
        upper = " 1 "
        
        ################    Part 4 - Tree generation   #############################
        # RF - Algorithm
        while (i < stop):
            gini_vector.fill(1)
            rd = random.sample(set(full_names[1:len(full_names)]), m_attributes)
            for element in range(1, branch[i].shape[1]):
                if len(branch[i]) == 0  and full_names[element] in rd:
                    skip_update = True 
                    break
                if (len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1) and full_names[element] in rd:
                     if "." not in rule[i]:
                         rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + "."
                         rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                         if i == 1 and (rule[i].find("{0}") != -1 or rule[i].find("{1}")!= -1):
                             rule[i] = rule[i].replace(".", "")
                     skip_update = True
                     break
                if is_number(dataset.iloc[:, element]) == True and full_names[element] in rd and bin_names[element] != "binary":
                    gini_vector[0, element] = 1.0
                    value = branch[i].iloc[:, element].unique()
                    skip_update = False
                    for bin_split in range(0, len(value)):
                        bin_sample = split_me(feature = branch[i].iloc[:, element], split = value[bin_split])
                        g_index = gini_index(target = branch[i].iloc[:, 0], feature = bin_sample[0], uniques = bin_sample[1])
                        if g_index < float(gini_vector[0, element]):
                            gini_vector[0, element] = g_index
                            uniqueWords[element] = bin_sample[1]
                    for rand in range (0, dataset.shape[1]):
                        if not full_names[rand] in rd:
                            gini_vector[0,rand] = 2.0
                if (is_number(dataset.iloc[:, element]) == False or bin_names[element] == "binary") and full_names[element] in rd:
                    gini_vector[0, element] = 1.0
                    skip_update = False
                    g_index = gini_index(target = branch[i].iloc[:, 0], feature =  pd.DataFrame(branch[i].iloc[:, element].values.reshape((branch[i].iloc[:, element].shape[0], 1))), uniques = uniqueWords[element])
                    gini_vector[0, element] = g_index
                    for rand in range (0, dataset.shape[1]):
                        if not full_names[rand] in rd:
                            gini_vector[0,rand] = 2.0
                       
            if skip_update == False:
                root_index = np.argmin(gini_vector)
                rule[i] = rule[i] + list(branch[i])[root_index]          
                for word in range(0, len(uniqueWords[root_index])):
                    uw = str(uniqueWords[root_index][word]).replace("<=", "")
                    uw = uw.replace(">", "")
                    lower = "<=" + uw
                    upper = ">" + uw
                    if uniqueWords[root_index][word] == lower and bin_names[root_index] != "binary":
                        branch.append(branch[i][branch[i].iloc[:, root_index] <= float(uw)])
                    elif uniqueWords[root_index][word] == upper and bin_names[root_index] != "binary":
                        branch.append(branch[i][branch[i].iloc[:, root_index]  > float(uw)])
                    else:
                        branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                    node = uniqueWords[root_index][word]
                    rule.append(rule[i] + " = " + "{" + str(node) + "}")            
                for logic_connection in range(1, len(rule)):
                    if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False  and rule[logic_connection].endswith("}") == True:
                        rule[logic_connection] = rule[logic_connection] + " AND "
            
            skip_update = False
            i = i + 1
            stop = len(rule)
        
        for i in range(len(rule) - 1, -1, -1):
            if rule[i].endswith(".") == False:
                del rule[i]   
        
        rule.append("Total Number of Rules: " + str(len(rule)))
        rule.append(dataset.agg(lambda x:x.value_counts().index[0])[0])
        
        if tree == 0:
            forest[tree] = rule
        else:
            forest.append(rule)

        print("Tree #", tree + 1, " was Planted")
    forest.append(ydata[0].unique()) 
    print("Forest is Fully Growth")
    return forest, oob_samples

    ############### End of Function ##############

######################## Part 5 - Usage ####################################

df = pd.read_csv('Python-DM-Classification-05-RF.csv', sep = ';')

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Building the Model
dt_model = dt_rf(X, y, forest_size = 5)

# Prediction
test =  df.iloc[:, 0:4]
pred = prediction_dt_rf(dt_model, test)

# Out-of-Bag Error Estimates
oob = oob_error_estimates(dt_model, X, y)
########################## End of Code #####################################
