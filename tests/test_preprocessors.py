from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from juniper.common import schema_tools
from juniper.common.data_type import FeatureType
from juniper.preprocessor.column_normalizer import ColumnNormalizer
from juniper.preprocessor.preprocessor import ColumnTransformer


def test_get_preprocessor(feature_store):
    preprocessor = ColumnTransformer(feature_store)
    assert preprocessor is not None
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 5
    assert preprocessor.transformers[0][0] == "numeric"
    assert preprocessor.transformers[1][0] == "categorical"
    assert preprocessor.transformers[2][0] == "boolean"
    assert preprocessor.transformers[3][0] == "timestamp"
    assert preprocessor.transformers[4][0] == "arr"


@pytest.mark.parametrize(
    "feature_type, expected",
    [
        (FeatureType.NUMERIC, [0.5154639, 0.0, -0.5154639, 0.0]),
        (FeatureType.CATEGORICAL, [0.0, 3.0, 1.0, 2.0]),
        (FeatureType.BOOLEAN, [1.0, 1.0, -1.0, 0.0]),
        (
            FeatureType.TIMESTAMP,
            np.array(
                [
                    [1.08805418e-02, 2.80798422e-02, 4.52708330e-02, 6.24484269e-02],
                    [7.81831482e-01, 9.74927912e-01, 4.33883739e-01, -4.33883739e-01],
                    [-6.01205130e-12, -1.71351267e-12, 2.58502595e-12, -7.66835066e-12],
                    [9.99940805e-01, 9.99605683e-01, 9.98974750e-01, 9.98048192e-01],
                    [6.23489802e-01, -2.22520934e-01, -9.00968868e-01, -9.00968868e-01],
                    [1.00000000e00, 1.00000000e00, 1.00000000e00, 1.00000000e00],
                ]
            ),
        ),
    ],
)
def test_fit_preprocessor(feature_type, expected, feature_store):
    column_transformer = ColumnTransformer(feature_store)
    name, preprocessor, columns = next(filter(lambda x: x[0] == feature_type, column_transformer.transformers))
    assert name == feature_type
    assert preprocessor is not None
    df = feature_store.read_parquet()
    Xt = preprocessor.fit_transform(df[columns])
    assert isinstance(Xt, pd.DataFrame)
    assert np.allclose(Xt.values.T, expected)


def test_get_array_metadata(feature_store):
    field = feature_store.schema.field("arr")
    cn = ColumnNormalizer(
        field=field, preprocessor_factory=partial(ColumnTransformer, feature_store=feature_store, prefix="arr.")
    )
    assert cn.schema_out is not None
    metadata = feature_store.get_feature_types(cn.schema_out)
    assert metadata is not None
    assert dict(metadata) == {
        FeatureType.NUMERIC: ["arr.a", "arr.b"],
    }


def test_normalize_array(feature_store):
    df = feature_store.read_parquet()[["arr"]]
    assert df is not None
    cn = ColumnNormalizer(
        field=feature_store.schema.field("arr"),
        preprocessor_factory=partial(ColumnTransformer, feature_store=feature_store, prefix="arr."),
    )
    Xt = cn._transform(df)
    assert Xt is not None
    assert isinstance(Xt, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "arr.a": [1.0, 3.0, 5.0, 7.0, 9.0],
            "arr.b": [2.0, 4.0, 6.0, 8.0, 10.0],
        },
        index=pd.Index([1, 1, 4, 4, 4], name="id"),
    )
    pd.testing.assert_frame_equal(Xt.astype(float), expected.astype(float))


def test_fit_array_preprocessor(feature_store):
    column_transformer = ColumnTransformer(feature_store)
    name, preprocessor, columns = next(filter(lambda x: x[0] == "arr", column_transformer.transformers))
    assert name == "arr"
    assert preprocessor is not None
    df = feature_store.read_parquet()
    Xt = preprocessor.fit_transform(df[columns])
    assert isinstance(Xt, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "arr": [
                np.array([[-0.5102040767669678, -0.2551020383834839], [-0.5102040767669678, -0.2551020383834839]]),
                np.array([[0.0], [0.0]]),
                np.array([[0.0], [0.0]]),
                np.array(
                    [[0.0, 0.2551020383834839, 0.5102040767669678], [0.0, 0.2551020383834839, 0.5102040767669678]]
                ),
            ],
        },
        index=pd.Index([1, 2, 3, 4], name="id"),
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_array_field_schema(feature_store):
    field = feature_store.schema.field("arr")
    schema = schema_tools.get_field_schema(field)
    expected = pa.schema(
        [
            pa.field("arr.a", pa.int64(), metadata={"usable_type": FeatureType.NUMERIC}),
            pa.field("arr.b", pa.int64(), metadata={"usable_type": FeatureType.NUMERIC}),
        ]
    )
    assert schema == expected

    schema = schema_tools.get_field_schema(
        field.with_metadata({"usable_type": f'[{{"a": "{FeatureType.UNUSABLE}", "b": "{FeatureType.UNUSABLE}"}}]'})
    )
    expected = pa.schema(
        [
            pa.field("arr.a", pa.int64(), metadata={"usable_type": FeatureType.UNUSABLE}),
            pa.field("arr.b", pa.int64(), metadata={"usable_type": FeatureType.UNUSABLE}),
        ]
    )
    assert schema == expected


@pytest.mark.parametrize(
    "feature_type, expected_n_transformers, expect_arr",
    [
        (FeatureType.NUMERIC, 5, True),
        (FeatureType.UNUSABLE, 4, False),
    ],
)
def test_array_unusable(feature_type, expected_n_transformers, expect_arr, feature_store):
    field = feature_store.schema.field("arr")
    schema = pa.schema(
        [feature_store.schema.field(name) for name in feature_store.schema.names if name != field.name]
    ).append(field.with_metadata({"usable_type": f'[{{"a": "{feature_type}", "b": "{FeatureType.UNUSABLE}"}}]'}))
    column_transformer = ColumnTransformer(feature_store, schema)
    assert len(column_transformer.transformers) == expected_n_transformers
    df = feature_store.read_parquet()
    Xt = column_transformer.fit_transform(df)
    assert Xt is not None
    assert FeatureType.NUMERIC in column_transformer.named_transformers_.keys()
    assert FeatureType.CATEGORICAL in column_transformer.named_transformers_.keys()
    assert FeatureType.BOOLEAN in column_transformer.named_transformers_.keys()
    assert FeatureType.TIMESTAMP in column_transformer.named_transformers_.keys()
    if expect_arr:
        assert "arr" in column_transformer.named_transformers_.keys()
    else:
        assert "arr" not in column_transformer.named_transformers_.keys()


def test_inference_all_null_values(feature_store):
    column_transformer = ColumnTransformer(feature_store)
    df = feature_store.read_parquet()
    column_transformer.fit(df)

    # create empty dataframe
    dat = df.head(1)
    dat = dat.apply(lambda x: [None] * len(x))
    assert all(pd.isna(dat))
    Xt = column_transformer.transform(dat)
    assert isinstance(Xt, pd.DataFrame)
    expected = pd.DataFrame(
        {
            "num": [0.0],
            # "missingindicator_num": [1.0],
            "cat": [3.0],
            "bool": [-1.0],
            "ts.year.sin": [0.0],
            "ts.week.sin": [0.0],
            "ts.day.sin": [0.0],
            "ts.year.cos": [1.0],
            "ts.week.cos": [1.0],
            "ts.day.cos": [1.0],
            "arr": [np.array([[0.0], [0.0]])],
        },
        index=pd.Index([1], name="id"),
    )

    pd.testing.assert_frame_equal(Xt, expected, check_dtype=False)
