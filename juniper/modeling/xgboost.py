import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

from juniper.modeling.metrics import SingleOutcomeEvalMetrics, EvalMetrics


class XGBClassifier(xgb.XGBClassifier):
    def metrics(self, x_test, y_test):
        scores = self.predict_proba(x_test)
        return EvalMetrics(
            epoch=1,
            metrics={
                "outcome": SingleOutcomeEvalMetrics(
                    roc_auc=roc_auc_score(y_test, scores),
                    log_loss=log_loss(y_test, scores),
                )
            },
        )

    def fit(self, x_train, y_train, x_test, y_test, **kwargs) -> EvalMetrics:
        super().fit(x_train, y_train)
        return self.metrics(x_test, y_test)
