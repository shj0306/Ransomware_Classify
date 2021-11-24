import os
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from data_process import DataProcess
from calc_auc import Calc_Auc


#성능 평가
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion

os.chdir('C:/sharedfolder/testmodel')#작업할 디렉토리 설정
test_software_path = '테스트용 소프트웨어 데이터.csv'
test_ransomware_path = '테스트용 랜섬웨어 데이터.csv'
model_name = "Doc2vec_model"

dp = DataProcess()
ca = Calc_Auc()

test_software_sentences, test_ransomware_sentences = dp.extract_data(test_software_path, test_ransomware_path)
test_data, test_labels = dp.make_vectordata(model_name, test_software_sentences, test_ransomware_sentences)
import xgboost as xgb

def test_model(test, features):

    dtest = xgb.DMatrix(test_data)
    import pickle
    loaded_model = pickle.load(open("XGboost_model.dat", "rb")) #학습시킨 모델 로드
    y_pred = loaded_model.predict(dtest)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)

    accuracy, precision, recall, _ = get_clf_eval(test_labels, y_pred)

    f1 = f1_score(test_labels, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 예측률
    print("Precision : %.2f%%" % (precision * 100.0)) #정밀도
    print("Recall : %.2f%%" % (recall * 100.0)) #재현율
    print("f1 score : %.2f%%" % (f1 * 100.0)) #f1_score

    ca.roc_curve_plot(test_labels, y_pred)

test_model(test_data, test_labels)
