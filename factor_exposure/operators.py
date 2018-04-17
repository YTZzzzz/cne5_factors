
import numpy as np
import pandas as pd

import datetime
import pickle








def winsorization_and_standardization(df_atomic_descriptor):

    #### Standardization ###

    std_df_atomic_descriptor = (df_atomic_descriptor - df_atomic_descriptor.mean()) / df_atomic_descriptor.std()


    #### Winsorization ###

    sd_atomic_descriptor = std_df_atomic_descriptor.std()

    mean_atomic_descriptor = std_df_atomic_descriptor.mean()

    upper_limit = mean_atomic_descriptor + 3* sd_atomic_descriptor

    lower_limit = mean_atomic_descriptor - 3 * sd_atomic_descriptor

    # Replace the outleirs

    std_df_atomic_descriptor[
        (std_df_atomic_descriptor > upper_limit) & (std_df_atomic_descriptor != np.nan)] = upper_limit

    std_df_atomic_descriptor[
        (std_df_atomic_descriptor < lower_limit) & (std_df_atomic_descriptor != np.nan)] = lower_limit

    return std_df_atomic_descriptor


def orthogonalize():

    return orthogolized_factor_exposure


