from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import time
from data_process import DataProcess
import os
from calc_auc import Calc_Auc

os.chdir('C:/sharedfolder/testmodel')#작업할 디렉토리 설정
train_software_path = '../software/behavior_api_order.csv'
train_malware_path = './behavior_api_order.csv'
test_software_path = './test_software_behavior_api_order.csv'
test_malware_path = './test_behavior_api_order.csv'
model_name = "../Doc2vec_model_vector30_window15_dm0"

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    return accuracy, precision, recall, confusion

dp = DataProcess()
ca = Calc_Auc()

train_software_sentences, train_malware_sentences = dp.extract_data(train_software_path, train_malware_path)
test_software_sentences, test_malware_sentences = dp.extract_data(test_software_path, test_malware_path)

x_train, y_train = dp.make_vectordata(model_name, train_software_sentences, train_malware_sentences)
x_test, y_test = dp.make_vectordata(model_name, test_software_sentences, test_malware_sentences)

gb_clf = GradientBoostingClassifier()
gb_model = gb_clf.fit(x_train, y_train)
import pickle
pickle.dump(gb_model,open("gbm_model.dat","wb")) #학습 모델 저장
loaded_model = pickle.load(open("gbm_model.dat","rb")) #학습 모델 로드

gb_pred = loaded_model.predict(x_test)
gb_pred = gb_pred > 0.5
gb_pred = gb_pred.astype(int)
gb_acc = accuracy_score(y_test, gb_pred)
acc, precision, recall, _ = get_clf_eval(y_test, gb_pred)
f1 = f1_score(y_test, gb_pred)

print("Accuracy: %.2f%%" % (acc * 100.0))  # 예측률
print("Precision : %.2f%%" % (precision * 100.0))  # 정밀도
print("Recall : %.2f%%" % (recall * 100.0))  # 재현율
print("f1 score : %.2f%%" % (f1 * 100.0))  # f1_score
ca.roc_curve_plot(y_test, gb_pred)
