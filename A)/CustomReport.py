from evidently.report import Report

class CustomReport(Report):
    def __init__(self, metrics, tags=None, metadata=None, options=None):
        super().__init__(metrics, options)
        self.tags = tags
        self.metadata = metadata

    def json(self):
        report_json = super().as_dict()
        # print(report_json)
        # del(report_json['metrics_preset'])
        report_json['metadata'] = self.metadata
        report_json['tags'] = self.tags
        return report_json

    
