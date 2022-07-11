import data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier


def train_model(filename):

    x_train, y_train, x_test, y_test = data.get_data(filename)

    clf_rf = RandomForestClassifier(class_weight={0: 1, 1: 100},
                                    min_samples_leaf=8,
                                    )

    clf_xgb = XGBClassifier(booster='gbtree',
                            gamma=0.01,
                            min_child_weight=1,
                            # as the data is highly imbalanced, leaf nodes can have smaller sized groups
                            max_delta_step=1,
                            objective='reg:squaredlogerror',
                            scale_pos_weight=100,  # to control the balance of pos and neg weight, = #neg/#pos
                            # predictor = 'gpu_predictor',
                            use_label_encoder=False,
                            verbosity=0
                            )

    estimator = [('XGBoost', clf_xgb), ('Random Forest', clf_rf)]

    clf = VotingClassifier(estimators=estimator, voting='soft')
    clf.fit(x_train, y_train)
    return clf
