import pandas as pd
from sklearn.impute import SimpleImputer


class ConstantImputer(SimpleImputer):
    def __init__(self, fill_value: any = None, add_indicator: bool = False, missing_values: any = pd.NA):
        super().__init__(
            strategy="constant", fill_value=fill_value, add_indicator=add_indicator, missing_values=missing_values
        )
