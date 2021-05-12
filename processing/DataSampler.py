
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class DataSampler:
    def __init__(self):
        pass

    def undersample(self, X, y, strategy='not minority') -> ('X', 'y'):
        '''Undersampling so all class sizes equal minority class size.'''
        rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
        X_undersampled, y_undersampled = rus.fit_resample(X, y)

        return X_undersampled, y_undersampled

    def oversample(self, X, y) -> ('X', 'y'):
        '''Oversample based on SMOTE so all class sizes equal majority class size.'''
        sm = SMOTE(random_state=42)
        X_oversampled, Y_oversampled = sm.fit_resample(X, y)

        return X_oversampled, Y_oversampled

    def sample_fixed_size(self, X, y, size: int) -> ('X', 'y'):
        sampling_sizes = {k: min(size, v) for k, v in y.value_counts().items()}

        # undersample the larger classes to size
        X, y = self.undersample(X, y, strategy=sampling_sizes)
        
        # oversample the smaller classes to size
        X, y = self.oversample(X, y)

        return X, y

    def sample_median_size(self, X, y: pd.Series) -> ('X', 'y'):
        '''Sample the median class size for all classes.'''
        median = int(y.value_counts().median())
        
        return self.sample_fixed_size(X, y, size=median)
