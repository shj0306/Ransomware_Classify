import csv
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from data_process import DataProcess


# tscne 그리는 함수.
def tscne_fun(model_name):
    '''
    model_name 으로 doc2vec 의 모델 이름을 넣어 주면됨.
    '''
    model = Doc2Vec.load(model_name)
    tags = list(model.docvecs.doctags.keys())  # dpcvecs에서 태그 데이터 가져옴.
    software_idx = []
    ransomware_idx = []
    for i, tag in enumerate(tags):
        # if tag.split('_')[0] == 'software':
        #   software_idx.append(i)#software의 배열 위치를 저장..
        if tag.split('_')[0] == 'ransomware':
            ransomware_idx.append(i)  # ransomware의 배열 위치를 저장.
    tsne = TSNE(n_components=2).fit(model.docvecs.doctag_syn0)  # 2차원으로 변환시킴
    datapoint = tsne.fit_transform(model.docvecs.doctag_syn0)
    fig = plt.figure()  # 특징 설정
    fig.set_size_inches(40, 20)  # 크기 셋팅
    ax = fig.add_subplot(1, 1, 1)  # subplot 생성

    # 악성코드 그리기. datapoint[ransomware_idx,0] x좌표,  datapoint[ransomware_idx,1] y 좌표
    ax.scatter(datapoint[ransomware_idx, 0], datapoint[ransomware_idx, 1], c='r')
    # 소프트웨어  그리기
    ax.scatter(datapoint[software_idx, 0], datapoint[software_idx, 1], c='b')
    fig.savefig(model_name + '.png')


'''
Doc2vec 형태의 모델 만들기
'''
os.chdir('C:/sharedfolder/test')  # 작업할 디렉토리 선택

train_software_path = '학습용 소프트웨어 데이터'
train_ransomware_path = '학습용 랜섬웨어 데이터'
model_name = "Doc2vec_model"

dp = DataProcess()

train_software_sentences, train_ransomware_sentences = dp.extract_data(train_software_path, train_ransomware_path)

data, labels = dp.make_vectordata(model_name, train_software_sentences,
                                  train_ransomware_sentences)  # 정상 소프트웨어와 랜섬웨어의 벡터화된 데이터를 합친 다음 data와 label을 따로 저장한다
'''
xgboost 학습 단계
'''
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

xgb.set_config(verbosity=0)

# 학습을 튜닝하기전에 먼저 Kfold로 일부를 나눈다.

kf_test = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf_test.split(data):
    # training set과 validation set을 나눈다
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]


# kFold로 학습시킨후 평균을 내어 반환 하는 함수.
# K개의 폴드를 만들어서 진행하는 교차 검증
# 총 데이터 갯수가 적은 데이터 셋에 대하여 정확도를 향상시킬 수 있다.

def kFoldValidation(train, features, xgbParams, numRounds, nFolds, target='loss'):  # 가장 마지막으로 호출됨.
    kf = KFold(n_splits=nFolds, shuffle=True)
    fold_score = []
    # train set, test set을 나눈다.
    for train_index, test_index in kf.split(train):
        X_train, X_valid = train[train_index], train[test_index]
        y_train, y_valid = features[train_index], features[test_index]
        # DMatrix : 넘파이 입력 파라미터를 받아서 만들어지는 XGBoost만의 전용 data set
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)
        # train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval' 로 명기
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        # early_stopping_rounds : 조기 중단을 위한 라운드를 설정합니다.
        # 조기 중단 기능 수행을 위해서는 반드시 eval_set과 eval_metric이 함께 설정되어야 합니다.
        gbm = xgb.train(xgbParams, dtrain, numRounds, evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=False)  # verbose 옵션으로 나타나지 않게함
        score = gbm.best_score
        fold_score.append(score)
    return np.mean(fold_score)


def xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample):
    # prepare xgb parameters
    params = {
        "objective": "reg:squarederror",  # reg:linear에서 변경됨
        "booster": "gbtree",  # gbtree : tree-based model, 일반적으로 가장 좋은 성능을 냄.
        "eval_metric": "mae", # mae : mean absolute error
        "tree_method": 'auto',
        "silent": 1,  # silent : 1 => 동작방식을 프린트하지 않음
        "eta": eta,  # learning rate [0,1]
        "max_depth": int(maxDepth),
        "min_child_weight": minChildWeight,  # over-fitting vs under-fitting을 조정하기 위한 파라미터
        "subsample": subsample,
        # 각 트리마다의 관측 데이터 샘플링 비율. 값을 적게 주면 over-fitting을 방지하지만 값을 너무 작게 주면 under-fitting이 발생할 수 있음.
        "colsample_bytree": colSample,  # 각 트리마다의 feature 샘플링 비율.
        "gamma": gamma,  # 분할을 수행하는데 필요한 최소 손실 감소를 지정한다.
        "scale_pos_weight": 1.3  # 데이터 셋 비율  0/1 소프트웨어 / 랜섬웨어
    }  # 적용할 파라미터들
    # 순서대로 train 학습시킬 데이터, features 특징, 기준이 될 xgb 파라미터, numRounds 반복횟수, nFolds
    cvScore = kFoldValidation(train, features, params, int(numRounds), nFolds=5)
    print('CV score: {:.6f}'.format(cvScore))
    return -1.0 * cvScore  # invert the cv score to let bayopt maximize

'''
Manual Search
- 최적의 하이퍼파라미터 값을 직접 탐색하는 방법
- 우리가 찾은 최적의 하이퍼파라미터 값이 실제로도 최적인지 보장하기가 어렵다

Grid Search
- Manual Search에 비해 Grid Search와 Random Search는 체계적인 방식으로 수행한다.
- Grid Search는 탐색의 대상이 되는 특정 구간 내의 후보 하이퍼 파라미터 값들을 
- 일정한 간격을 두고 선정하여, 성능 결과를 기록한 뒤 가장 높은 성능을 냈던 값을 선정하는 방법
- 좀 더 균등하고 전역적인 탐색이 가능하다
- 하이퍼파라미터의 개수를 한번에 여러 종류로 가져갈수록 탐색시간이 증가함.

Random Search
- 탐색 대상 구간 내의 후보 하이퍼파라미터 값들을 랜덤 샘플링을 통해 선정한다.
- 랜덤 서치는 모든 그리드를 전부 보지 않고 랜덤하게 일부의 파라미터들만 관측한 후 그 중 가장 
  좋은 파라미터를 고른다.
- 그리드 서치와 랜덤 서치 모두 다음 번 시도할 후보 하이퍼파라미터 값을 선정하는 과정에서
- 이전까지 조사 과정에서 얻어진 하이퍼파라미터 값들의 성능 결과에 대한 사전 지식이 반영되어 있지 않기 때문이다.

베이지안 최적화는 어느 입력값을 받는 미지의 목적 함수를 상정하여,
해당 함숫값을 최대로 만드는 최적해를 찾는 것을 목적으로 합니다.

목적 함수와 하이퍼파라미터 쌍을 대상으로 대체 모델을 만들고,
순차적으로 하이퍼파라미터를 업데이트해 가면서 평가를 통해 최적의 하이퍼파라미터 조합을 탐색합니다.
이 때 목적 함수를 블랙박스 함수라고 합니다.
'''


def bayesOpt(train, features):  # 가장먼저 호출됨.
    ranges = {
        'numRounds': (1000, 2000),
        'eta': (0.03, 0.1),
        'gamma': (0, 10),
        'maxDepth': (4, 10),
        'minChildWeight': (0, 10),
        'subsample': (0, 1),
        'colSample': (0, 1),
    }  # 학습에 따라 변경될 값들.
    # proxy through a lambda to be able to pass train and features
    optFunc = lambda numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample: xgbCv(train, features,
                                                                                                  numRounds, eta, gamma,
                                                                                                  maxDepth,
                                                                                                  minChildWeight,
                                                                                                  subsample, colSample)
    bo = BayesianOptimization(optFunc, ranges)
    # verbose : 0 => silent, 1 => prints only when a maximum is observed 2 => always print(default)

    bo.maximize(init_points=50, n_iter=10, kappa=2, acq="ei", xi=0.0)  # maximize하는 방향으로 hyper-parameter값을 계산
    best_params = max(bo.res, key=lambda x: x['target'])['params']

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    booster = xgb.train(best_params, dtrain, num_boost_round=1980)
    dtest = xgb.DMatrix(test_data)
    y_pred = booster.predict(dtest)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 예측률

    import pickle
    pickle.dump(booster, open("XGboost_model.dat", "wb"))  # 학습 모델 저장
    loaded_model = pickle.load(open("XGboost_model.dat", "rb"))  # 학습 모델 로드
    y_pred = loaded_model.predict(dtest)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_pred, test_labels)
    print("model Accuracy: %.2f%%" % (accuracy * 100.0))  # 예측률


bayesOpt(data, labels)
