# Juniper

Juniper is a library containing a few helper functions for working with panel data. It contains three main classes:

1. `juniper.preprocessor.ColumnTransformer` is an extension of the sklearn ColumnTransformer with a friendly init method. It will automatically infer the column types from a pyarrow schema and apply the appropriate transformers to the columns.
2. `juniper.preprocessor.ColumnNormalizer`: This class essentially takes a pandas Series and applies [pd.json_normalize](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html) to it, followed by a ColumnTransformer.
3. `juniper.modeling.layers.Unify` is a pytorch layer that will apply the seq2vec model(s) of your choice to any number of jagged panel data columns, then concatenate the results. See `juniper.modeling.layers.SummaryPool` for a simple example.

To use this library, ensure that your dataset can produce a valid `pyarrow.Schema` with the following metadata on each field:
- `usable_type`: see `data_type.py` for a list of valid types
- `record_path` and `meta` for each column that will be passed to `ColumnNormalizer`