from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def RF_test(model, pca, X_test):
    X_test = pca.transform(X_test)
    predictions = model.predict(X_test)
    return predictions


def RF_train(X_train, y_train, n_estimators, max_depth, min_samples_leaf):
    model_RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)  # , max_features = 4)#, class_weight = 'balanced')
    model_RF.fit(X_train, y_train)
    train_predictions = model_RF.predict_proba(X_train)
    return model_RF, train_predictions[:, 1]


def CV(X, y, k_fold, n_estimators, max_depth, min_samples_leaf, n_components):
    cv_AUC = []
    train_AUC = []
    model = None
    pca = None
    for i in range(k_fold):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=round(1 / k_fold, 2))
        # scaler_x = StandardScaler()
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        # X_train  = scaler_x.fit_transform(X_train)
        # X_val    = scaler_x.transform(X_val)
        model, train_predictions = RF_train(X_train, y_train, n_estimators, max_depth, min_samples_leaf)
        train_AUC.append(roc_auc_score(y_train, train_predictions))
        val_predictions = model.predict_proba(X_val)
        cv_metric = roc_auc_score(y_val, val_predictions[:, 1])
        cv_AUC.append(cv_metric)
    print('Training   AUC  =', np.mean(np.array(train_AUC)))
    return np.mean(np.array(cv_AUC)), model, pca


def RF_run(X, y, k_fold=10, n_components=4, n_models=1):
    print('RANDOM FOREST')
    print("N_components  = ", n_components)
    best_AUC = 0
    best_model = None
    pca = None
    for i in range(n_models):
        n_estimators = np.random.randint(1, 20)     #11
        max_depth = np.random.randint(5, 10)        #5
        min_samples_leaf = np.random.randint(3, 10) #4
        n_estimators = 10 * n_estimators
        print("N_estimators       =", n_estimators)
        print("Max_depth          =", max_depth)
        print("Min sample leaf    =", min_samples_leaf)
        cv_AUC, model, pca = CV(X, y, k_fold, n_estimators, max_depth, min_samples_leaf, n_components)
        print("Cross val  AUC  =", cv_AUC)
        if cv_AUC >= best_AUC:
            best_model = model
            best_AUC = cv_AUC
    return best_model, pca

