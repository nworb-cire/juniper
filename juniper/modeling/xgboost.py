import onnx
from onnxconverter_common import FloatTensorType
from onnxmltools.convert import convert_xgboost

import xgboost
from juniper.common.component import ModelComponent
from juniper.common.export import merge_models, add_metrics
from juniper.common.schema_tools import get_input_mapping
from juniper.modeling.metrics import EvalMetrics
from juniper.preprocessor.preprocessor import ColumnTransformer


class XGBClassifier(ModelComponent):
    def __init__(self, *, preprocessor: ColumnTransformer, **kwargs):
        self.preprocessor = preprocessor
        self.preprocessor_onnx = self.preprocessor.to_onnx()
        self.preprocessor_inputs = get_input_mapping(self.preprocessor)
        self.clf = xgboost.XGBClassifier(**kwargs)

    def fit(self, *args, **kwargs):
        self.clf.fit(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.clf.predict_proba(*args, **kwargs)

    def to_onnx(self, name: str | None = None, metrics: list[EvalMetrics] | None = None) -> onnx.ModelProto:
        input_size = self.preprocessor_onnx.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        input_name = self.preprocessor_onnx.graph.input[0].name
        model = convert_xgboost(self.clf, initial_types=[(input_name, FloatTensorType([1, input_size]))])
        io_map = [(input_name, input_name)]
        merged = merge_models(self.preprocessor_onnx, model, io_map=io_map)
        onnx.checker.check_model(merged, full_check=True)

        # add_default_metadata(merged)
        if metrics is not None:
            add_metrics(merged, metrics)
        return merged

    def validate(self, model: onnx.ModelProto):
        pass
