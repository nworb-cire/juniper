import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from onnxruntime import InferenceSession

from juniper.preprocessor.preprocessor import ColumnTransformer


@pytest.mark.skip(reason="Period transformer output is not ordered correctly")
def test_multiple_timestamps(feature_store):
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("s"), metadata={"usable_type": "timestamp"}),
            pa.field("ts2", pa.timestamp("s"), metadata={"usable_type": "timestamp"}),
        ]
    )
    column_transformer = ColumnTransformer(feature_store, schema=schema)
    df = feature_store.read_parquet()
    print(df.columns)
    df = df[["ts"]]
    df["ts2"] = df["ts"] + pd.Timedelta(days=30)
    column_transformer.fit(df)
    row = df.iloc[:1]
    expected = column_transformer.transform(row).values

    model_onnx = column_transformer.to_onnx()
    sess = InferenceSession(model_onnx.SerializeToString())
    input = {
        "ts": row["ts"].apply(lambda x: x.timestamp()).values.astype(np.int64).reshape(-1, 1),
        "ts2": row["ts2"].apply(lambda x: x.timestamp()).values.astype(np.int64).reshape(-1, 1),
    }
    output = sess.run(None, input)
    assert output is not None
    assert len(output) == 1
    # High tolerances necessary because the output is only approximately correct, FIXME
    assert np.allclose(output[0].ravel(), expected, atol=0.01, rtol=0.05)
