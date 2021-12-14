class RepeatedTrainingResult:

    def __init__(self):
        self.classification_reports = []

    def add_classification_report(self, report: dict):
        self.classification_reports.append(report)

    def get_avg_accuracy(self):
        avg_acc = 0
        for report in self.classification_reports:
            avg_acc += report['accuracy']

        return avg_acc / len(self.classification_reports)

    def get_avg_metric(self, report_key1: str, report_key2: str):
        '''Returns the average values for key1=macro/weighted key2=precision/recall/f1-score'''
        avg_val = 0
        for report in self.classification_reports:
            avg_val += report[report_key1+" avg"][report_key2]

        return avg_val / len(self.classification_reports)

    def get_all_metrics(self) -> dict:
        return {
            'accuracy': self.get_avg_accuracy(),
            'macro avg': {
                'precision': self.get_avg_metric('macro', 'precision'),
                'recall': self.get_avg_metric('macro', 'recall'),
                'f1-score': self.get_avg_metric('macro', 'f1-score'),
            },
            'weighted avg': {
                'precision': self.get_avg_metric('weighted', 'precision'),
                'recall': self.get_avg_metric('weighted', 'recall'),
                'f1-score': self.get_avg_metric('weighted', 'f1-score'),
            },
        }
    
    def get_all_metrics_as_str(self) -> str:
        return \
        f"{self.get_avg_accuracy()}," \
        f"{self.get_avg_metric('macro', 'precision')}," \
        f"{self.get_avg_metric('macro', 'recall')}," \
        f"{self.get_avg_metric('macro', 'f1-score')}," \
        f"{self.get_avg_metric('weighted', 'precision')}," \
        f"{self.get_avg_metric('weighted', 'recall')}," \
        f"{self.get_avg_metric('weighted', 'f1-score')}," 
        

# res = RepeatedTrainingResult()
# res.add_classification_report({"-1.0": {"precision": 0.2, "recall": 1.0, "f1-score": 0.33333333333333337, "support": 1}, "3.0": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 4}, "accuracy": 0.2, "macro avg": {"precision": 0.1, "recall": 0.5, "f1-score": 0.16666666666666669, "support": 5}, "weighted avg": {"precision": 0.04, "recall": 0.2, "f1-score": 0.06666666666666668, "support": 5}})
# print(res.get_all_metrics_as_str())
