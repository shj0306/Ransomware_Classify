from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

class Calc_Auc():
    def roc_curve_plot(self, y_test, y_pred):
        # 임계값에 따른 FPR, TPR 값
        fprs, tprs, thresholds = roc_curve(y_test, y_pred)
        # ROC 곡선을 시각화
        plt.plot(fprs, tprs, label='ROC')
        # 가운데 대각선 직선
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        # FPR X 축의 scla 0.1 단위 지정
        start, end = plt.xlim()
        plt.xticks(np.round(np.arange(start, end, 0.1), 2))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('FPR( 1 - Sensitivity )')
        plt.ylabel('TPR( Recall )')
        plt.text(0.1, 0.8, roc_auc_score(y_test, y_pred))
        plt.legend()
        plt.show()