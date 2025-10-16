# Standard
from collections import Counter
from typing import Any, List, Literal, Tuple, Union
import random
import warnings

# Third Party
from numpy import linalg as LA
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats

# Local
from fms_dgt.utils import dgt_logger

# ===========================================================================
#                        PREPROCESSING BLOCK
# ===========================================================================


def preprocess_train_data(
    data,
    train_channels,
    train_length,
    train_samples,
    augmentation_strategy,
    min_windows_length,
    min_windows_number=10,
    train_splitting="minimize-overlap",
):
    if augmentation_strategy in ("multisample"):
        return preprocess_multisample_data(data, train_channels, train_samples, train_length)
    elif augmentation_strategy in ("univariate", "multivariate"):
        return preprocess_uni_multi_variate_data(
            data,
            train_channels,
            train_length,
            min_windows_length,
            min_windows_number,
            train_splitting,
        )
    else:
        raise ValueError(
            f"Unsupported augmentation strategy: '{augmentation_strategy}'. "
            "Choose from 'multisample', 'univariate', or 'multivariate'."
        )


def preprocess_uni_multi_variate_data(
    data, train_channels, train_length, min_windows_length, min_windows_number, train_splitting
):
    if train_splitting not in {"minimize-overlap", "maximize-overlap", "no-periodicity"}:
        raise ValueError(
            f"Unsupported train_splitting strategy: '{train_splitting}'. "
            "Expected one of: 'minimize-overlap', 'maximize-overlap', 'no-periodicity'."
        )

    start = 0
    end_train = train_length
    train_data = get_data(data, start, end_train, train_channels)

    preprocessed_train = []
    chosen_period_per_channel = []

    # Iterate over each channel and calculate its period and window length
    for idx, channel_data in enumerate(train_data):
        channel_name = train_channels[idx]
        dgt_logger.info("CHANNEL: %s", channel_name)

        chosen_period, _ = compute_period(
            channel_data, min_points=min_windows_length, min_windows=min_windows_number
        )
        dgt_logger.info("Chosen period %d for channel (%s)", chosen_period, channel_name)
        chosen_period_per_channel.append(chosen_period)

    # Use the most frequent period across channels
    chosen_period = max(chosen_period_per_channel, key=chosen_period_per_channel.count)
    dgt_logger.info("Overall chosen period: %d", chosen_period)

    # Construct windows for each channel using the chosen period
    for idx, channel_data in enumerate(train_data):
        verbose = idx == 0
        windows, _ = construct_windows(
            channel_data,
            chosen_period,
            min_windows_length,
            min_windows_number,
            train_splitting,
            verbose,
        )
        preprocessed_train.append(windows)

    # Scale the preprocessed data
    scaled_preprocessed_train, list_fitted_scaler = standardscale_train(preprocessed_train)

    return (scaled_preprocessed_train, preprocessed_train, list_fitted_scaler)


def preprocess_multisample_data(data, train_channels, train_samples, train_length):

    sample_str = train_channels[0]
    sample_columns = [col for col in data.columns if col.startswith(sample_str)]

    if len(list(sample_columns)) <= train_samples:
        dgt_logger.info("Enough samples, choosing windows in different samples")
        raise ValueError(
            f"Number of samples ({len(list(sample_columns))}) "
            f"is less than "
            f"the total number of samples wanted ({train_samples})."
        )

    id_samples = random.sample(list(sample_columns), train_samples)

    start = 0
    end = train_length

    # train data
    id_train_samples = id_samples[0:train_samples]
    train_data = get_data(data, start, end, id_train_samples)
    train_data = [train_data.tolist()]

    scaled_preprocessed_train, list_fitted_scaler = standardscale_train(train_data)

    return (scaled_preprocessed_train, train_data, list_fitted_scaler)


def get_feature_distribution(column: List[Any]) -> Tuple[str, bool, Union[dict, Counter]]:
    if isinstance(column[0], float):
        _, p_value = stats.shapiro(column)
        normal_distr = p_value > 0.05
        stats_summary = {
            "min": min(column),
            "max": max(column),
            "mean": np.mean(column),
            "median": np.median(column),
            "std": np.std(column),
            "data": column,
        }
        return "numerical", normal_distr, stats_summary
    else:
        return "categorical", False, Counter(column)


# ===========================================================================
#                        Data upload and preprocessing
# ===========================================================================


def get_data(data, start, end, id_channel):

    indices_channel = [data.columns.get_loc(channel) for channel in id_channel]
    data = data.T

    if start < 0 or end > data.shape[1]:
        raise ValueError(
            f"Not enough data present!!! Trying to get data from indices:{start} to {end} but available data is {data.shape[1]} "
        )

    train_data = data.to_numpy()[indices_channel, start:end]
    train_data = np.array(train_data, dtype=np.float32)
    train_data = interpolate_na(train_data)

    return train_data


def interpolate_na(data):
    """
    Interpolates NaN values in each channel of the train data using linear interpolation.

    Args:
        data (np.array): 2D numpy array of shape (n_channels, n_points),
                               where each row is a time series with NaN values.

    Returns:
        np.array: 2D numpy array with NaN values interpolated.
    """
    interpolated_data = []

    for channel in data:
        # Convert to a Pandas Series to use interpolate method
        channel_series = pd.Series(channel)
        # Interpolate NaN values linearly
        channel_interpolated = channel_series.interpolate(method="linear", limit_direction="both")
        # Append the interpolated channel back to list
        interpolated_data.append(channel_interpolated.values)

    # Convert the list back to a numpy array
    return np.array(interpolated_data)


def standardscale_train(preprocessed_train):
    # Standard Scaling the data useful for aggregated scores in evaluation

    # -- by timestamps --
    scaled_preprocessed_train = []
    list_fitted_scaler = []

    for idx, channel_data in enumerate(preprocessed_train):
        arr_train = np.array(channel_data)
        scaler = StandardScaler()
        scaled_arr_train = scaler.fit_transform(arr_train)

        scaled_preprocessed_train.append(scaled_arr_train)
        list_fitted_scaler.append(scaler)

        dgt_logger.info(
            "Channel %d train data standardized. Shape: %s", idx, scaled_arr_train.shape
        )

    return np.array(scaled_preprocessed_train), list_fitted_scaler


# ===========================================================================
#                       Periodicity Aware Segmentation
# ===========================================================================


def estimate_period_acf(time_series):
    """
    Estimate the dominant periods in a time series based on its autocorrelation function (ACF).

    Parameters:
    time_series (array-like): The input time series data.

    Returns:
    list: Lags corresponding to the most significant peaks in the ACF, sorted by significance.
    """
    # Compute the ACF of the time series
    autocorr = acf(time_series, nlags=len(time_series) // 2, fft=True)

    # Identify peaks in the ACF, excluding lag 0
    peaks, _ = find_peaks(autocorr[1:])  # Skip lag 0
    peaks += 1  # Adjust indices since lag 0 was skipped

    # Evaluate and sort peaks by their autocorrelation values
    sorted_peaks = sorted(peaks, key=lambda x: autocorr[x], reverse=True)

    return sorted_peaks


def compute_period(time_series, min_points=1120, min_windows=10):
    """
    Computes the period of a time series using ACF as the primary method
    and Fourier Transform as a fallback. Ensures the period allows for at
    least `min_windows` windows and meets the `min_points` condition.

    Parameters:
        time_series (array-like): The input time series data.
        min_points (int): Minimum points required for each window.
        min_windows (int): Minimum number of windows required.

    Returns:
        period (int): The computed period for the time series.
        window_length (int): The calculated window length.
    """
    # Determine maximum allowable period
    max_period = max(1, (len(time_series) - min_points) // (min_windows - 1))

    # Try to find the period using ACF
    acf_periods = estimate_period_acf(time_series)
    period = max_period
    period_source = "Fallback to max_period"
    position = len(acf_periods)

    for idx, p in enumerate(acf_periods):
        if p < max_period:
            period = p
            period_source = "ACF"
            position = idx
            break

    # If no valid period is found using ACF, fallback to Fourier Transform
    if period_source != "ACF":
        freqs = scipy.fftpack.fftfreq(len(time_series), d=1)
        power_spectrum = np.abs(scipy.fftpack.fft(time_series)) ** 2
        positive_freq_indices = np.where(
            (freqs > 0) & (1 / np.where(freqs != 0, freqs, np.inf) <= max_period)
        )[0]

        if positive_freq_indices.size > 0:
            peak_index = positive_freq_indices[np.argmax(power_spectrum[positive_freq_indices])]
            peak_freq = freqs[peak_index]
            period = round(1 / peak_freq)
            period_source = "FFT"

    # Calculate window length based on the determined period
    N = max(1, (min_points + period - 1) // period)
    window_length = N * period

    # Diagnostics
    dgt_logger.info("  - Max possible period: %s", max_period)
    dgt_logger.info("  - First ACF periods (ranked): %s", acf_periods[: (position + 1)])
    dgt_logger.info("  - Selected period: %s (Source: %s)", period, period_source)

    return period, window_length


def construct_windows(
    time_series, period, window_length=1120, min_windows=10, split="maximize-overlap", verbose=True
):
    """
    Constructs windows for the time series based on the computed period.
    Ensures that the overlapping points of the last window are a multiple of the period.
    """

    # Try to compute the number of windows based on minimal overlap
    computed_windows = len(time_series) // window_length

    windows = []
    start = 0
    extra_window = False
    overlap_extra_window, periods_extra_window = None, None

    if split == "no-periodicity":
        # Compute total span of data and spacing to get min_windows
        max_start = len(time_series) - window_length
        if min_windows <= 1 or max_start <= 0:
            windows.append(time_series[:window_length])
            overlap_step = 0
        else:
            overlap_step = max_start // (min_windows - 1)
            for i in range(min_windows):
                start = i * overlap_step
                end = start + window_length
                if end <= len(time_series):
                    windows.append(time_series[start:end])
                else:
                    # If the last window goes out of bounds, adjust
                    windows.append(time_series[-window_length:])
                    break
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step

    elif split == "maximize-overlap":
        overlap_step = period
        # Generate initial windows
        while start + window_length <= len(time_series):
            windows.append(time_series[start : start + window_length])
            start += overlap_step
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step

    elif split == "minimize-overlap":
        # Calculate overlap_step
        if computed_windows >= min_windows:
            number_full_periods_in_windows = window_length // period
            overlap_step = number_full_periods_in_windows * period

            # Generate initial windows
            while start + window_length <= len(time_series):
                windows.append(time_series[start : start + window_length])
                start += overlap_step
        else:
            # Calculate overlap step to get exactly min_windows
            max_start_pos = len(time_series) - window_length
            overlap_step = max(1, (max_start_pos + (min_windows - 1)) // (min_windows - 1))

            # Ensure overlap_step is a multiple of the period
            overlap_step = (overlap_step // period) * period

            # Generate windows with exact min_windows count
            windows = []
            start = 0
            for _ in range(min_windows):
                if start + window_length > len(
                    time_series
                ):  # Adjust last window if exceeding bounds
                    start = len(time_series) - window_length
                windows.append(time_series[start : start + window_length])
                start += overlap_step

        # Calculate total coverage and check if we can add an extra window
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step
        if len(time_series) - total_coverage > period:
            # Add an extra window to maximize total coverage
            extra_window = True
            end_extra_windows = (
                total_coverage + ((len(time_series) - total_coverage) // period) * period
            )
            windows.append(time_series[end_extra_windows - window_length : end_extra_windows])
            overlap_extra_window = len(windows[0]) - (end_extra_windows - total_coverage)
            periods_extra_window = overlap_extra_window // period
            total_coverage += len(windows[0]) - overlap_extra_window

    else:
        raise ValueError(
            "Unknown SPLIT parameters. Possible values: maximize-overlap, minimize-overlap, no-periodicity"
        )

    # Diagnostics
    if verbose:
        dgt_logger.info("Period: %s", period)
        dgt_logger.info("Number of windows: %s", len(windows))
        dgt_logger.info("Points in each window: %s", window_length)
        dgt_logger.info("Periods in a window: %s", window_length // period)
        dgt_logger.info("Overlapping points: %s", window_length - overlap_step)
        dgt_logger.info("Overlapping periods: %s", (window_length - overlap_step) // period)
        if extra_window:
            dgt_logger.info("Extra window overlapping points: %s", overlap_extra_window)
            dgt_logger.info("Extra window overlapping periods: %s", periods_extra_window)
            dgt_logger.info("Total coverage: %s\n", total_coverage)

    return windows, window_length - overlap_step


# ===========================================================================
#                       EMBEDDING BLOCK
# ===========================================================================


def fpc_embed_data(preprocessed_data, embedding_dim, percentage_of_variance_explained):

    n_var = len(preprocessed_data)  # Number of features

    # Compute embeddings
    data_embedded = []
    data_embedded_full = []
    fpc_basis = []
    fpc_basis_full = []
    var_explained = []
    embedding_dims = []
    full_embedding_dims = []
    sdforger_input = []

    for var in range(n_var):
        # Standardize each feature independently
        data_train_std = preprocessed_data[
            var
        ]  # Standardization already done in preprocessing step

        # Compute FPCA with dynamic embedding_dim based on percentage_of_variance
        eigvals_var, eigenfuns = LA.eigh(data_train_std.T @ data_train_std)
        eigvals_sorted = eigvals_var[::-1]
        cumsum_eigvals = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted)

        # Determine embedding dimensions
        if embedding_dim == "auto":
            embedding_dim = max(
                2, np.argmax(cumsum_eigvals >= percentage_of_variance_explained) + 1
            )

        else:
            embedding_dim = embedding_dim  # Use fixed embedding dimension if provided

        dgt_logger.info(
            "Latent dimension channel %s: %s",
            var,
            embedding_dim,
        )
        dgt_logger.info(
            "\nVariance retained for channel %s: %.4f",
            var,
            cumsum_eigvals[embedding_dim - 1],
        )
        full_embedding_dim = max(np.argmax(cumsum_eigvals >= 0.99) + 1, embedding_dim)
        embedding_dims.append(embedding_dim)
        full_embedding_dims.append(full_embedding_dim)

        # Store eigenfunctions and truncated eigenvalues
        data_embedded.append((data_train_std @ eigenfuns[:, -embedding_dim:]).copy())
        data_embedded_full.append((data_train_std @ eigenfuns[:, -full_embedding_dim:]).copy())
        fpc_basis.append(eigenfuns[:, -embedding_dim:])
        fpc_basis_full.append(eigenfuns[:, -full_embedding_dim:])
        var_explained.append(cumsum_eigvals[embedding_dim - 1])

        # Prepare SDForger input
        sdforger_input.append(data_embedded[-1])

    sdforger_input = np.hstack(sdforger_input)

    return (
        pd.DataFrame(
            sdforger_input,
            columns=["value_" + str(k) for k in range(0, sdforger_input.shape[1])],
        ),
        embedding_dims,
        data_embedded,
        data_embedded_full,
        fpc_basis,
        fpc_basis_full,
    )


def fica_embed_data(preprocessed_data, embedding_dim, percentage_of_variance_explained):

    n_var = len(preprocessed_data)  # Number of features

    # Compute embeddings
    fica_mixing = []
    fica_mean = []
    data_embedded = []
    embedding_dims = []
    sdforger_input = []
    var_explained = []

    for var in range(n_var):
        data_train_std = preprocessed_data[var]

        # Determine optimal k (number of components)
        if embedding_dim == "auto":
            k_auto = 2
            max_k = 30
            while k_auto <= max_k:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fica = FastICA(n_components=k_auto, random_state=0)
                    data_temp = fica.fit_transform(data_train_std)
                    data_reconstructed = data_temp @ fica.mixing_.T + fica.mean_

                # Compute total variance retained across all samples and features
                original_var = np.var(data_train_std, axis=1).sum()
                reconstructed_var = np.var(data_reconstructed, axis=1).sum()
                retained = reconstructed_var / original_var

                if retained >= percentage_of_variance_explained:
                    break
                k_auto += 1
            embedding_dim = k_auto
        else:
            embedding_dim = embedding_dim

        dgt_logger.info(
            "Latent dimension channel %s: %s",
            var,
            embedding_dim,
        )
        fica = FastICA(n_components=embedding_dim, random_state=0)
        data_embedded.append(fica.fit_transform(data_train_std))
        data_reconstructed = data_embedded[-1] @ fica.mixing_.T + fica.mean_

        # Compute retained variance (final report)
        original_var = np.var(data_train_std, axis=1).sum()
        reconstructed_var = np.var(data_reconstructed, axis=1).sum()
        retained = reconstructed_var / original_var
        dgt_logger.info(
            "\nVariance retained for channel %s: %.4f",
            var,
            retained,
        )

        embedding_dims.append(embedding_dim)
        fica_mixing.append(fica.mixing_)
        fica_mean.append(fica.mean_)
        sdforger_input.append(data_embedded[-1])
        var_explained.append(retained)

    sdforger_input = np.hstack(sdforger_input)

    return (
        pd.DataFrame(
            sdforger_input,
            columns=["value_" + str(k) for k in range(0, sdforger_input.shape[1])],
        ),
        embedding_dims,
        data_embedded,
        fica_mixing,
        fica_mean,
    )


def embeddings_to_text(
    row: Any,
    columns: List[str],
    eos_token: str,
    permute: bool = True,
    text_template: Literal[
        "base_template", "fim_template", "fim_template_textual_encoding"
    ] = "fim_template",
    input_tokens_precision: int = 4,
):

    # Shuffle columns, if requested
    if permute:
        random.shuffle(columns)

    decimal = input_tokens_precision

    # Prepare text based on the requested template
    if text_template == "base_template":
        text = ", ".join(
            [
                (
                    f"{col} is {row[col]:.{decimal}f}"
                    if isinstance(row[col], float)
                    else f"{col} is {row[col]}"
                )
                for col in columns
            ]
        )
        text = text + eos_token
    elif text_template == "fim_template":
        text_input = ", ".join([f"{col} is [blank]" for col in columns])
        text_target = " ".join(
            [
                (
                    f"{row[col]:.{decimal}f} [answer]"
                    if isinstance(row[col], float)
                    else f"{row[col]} [answer]"
                )
                for col in columns
            ]
        )
        text = "Input: " + text_input + " [sep] Target: " + text_target + eos_token
    else:
        text_categorical = "".join(
            [
                (f"{col} is {row[col]}, " if not isinstance(row[col], float) else "")
                for col in columns
            ]
        ).strip(", ")

        text_input = "".join(
            [(f"{col} is [blank], " if isinstance(row[col], float) else "") for col in columns]
        ).strip(", ")
        text_target = "".join(
            [
                (f"{float(row[col]):.{decimal}f} [answer] " if isinstance(row[col], float) else "")
                for col in columns
            ]
        ).strip(" ")
        text = (
            "Condition: "
            + text_categorical
            + " [sep] Input: "
            + text_input
            + " [sep] Target: "
            + text_target
            + eos_token
        )

    # Return
    return text


# ===========================================================================
#                      RECOVERING BLOCK
# ===========================================================================


def convert_texts_to_tabular_data(
    text: List[str],
    original_dataset: pd.DataFrame,
    text_template: Literal[
        "base_template", "fim_template", "fim_template_textual_encoding"
    ] = "fim_template",
) -> pd.DataFrame:
    """Converts the sentences back to tabular data

    Args:
        text (List[str]): List of the tabular data in text form
        original_dataset (pd.DataFrame): Original dataset
        text_template (str): Text template

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """

    feature_distribution = {
        col: get_feature_distribution(original_dataset[col].tolist())
        for col in original_dataset.columns
    }
    columns = original_dataset.columns.to_list()

    generated = []

    if text_template == "base_template":
        # Convert text to tabular data for base_template
        for t in text:
            features = t.split(",")
            td = dict.fromkeys(columns, "NaN")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(" is ")
                if values[0] in columns and td[values[0]] == "NaN":
                    try:
                        if len(values) >= 2:
                            if feature_distribution[values[0]][0] == "numerical":
                                i = 1
                                while i <= len(values):
                                    try:
                                        td[values[0]] = values[i]
                                        i = len(values) + 1
                                    except ValueError:
                                        i += 1
                                        continue
                            elif feature_distribution[values[0]][0] == "categorical":
                                # TODO: create categorical check
                                pass
                    except IndexError:
                        pass
            generated.append(td)
        df_gen = pd.DataFrame(generated)
        df_gen.replace("None", None, inplace=True)

    if text_template == "fim_template":
        # Convert text to tabular data
        for t in text:
            try:
                features_input = t.split(",")
                td = dict.fromkeys(columns, "NaN")

                input_part, output_part = t.split(" [sep] Target: ")
                input_values = input_part.removeprefix("Input: ")
                output_values = output_part.removesuffix(" [answer]")

                features_input = input_values.split(",")
                features_output = output_values.split(" [answer] ")
            except ValueError:
                # If text is malformed, skip it
                continue
            if len(features_output) >= len(features_input):
                # Transform all features back to tabular data
                for i, f in enumerate(features_input):
                    col_i = f.strip().split(" is ")[0]
                    if col_i in columns:
                        try:
                            val_i = features_output[i]
                            if col_i in columns:
                                if feature_distribution[col_i][0] == "numerical":
                                    td[col_i] = float(val_i)  # Parse numerical values
                                else:
                                    td[col_i] = val_i  # Keep categorical as string
                        except ValueError:
                            continue
            generated.append(td)
        # Create DataFrame and replace placeholder "NaN" with actual NaN
        df_gen = pd.DataFrame(generated)
        df_gen.replace("NaN", None, inplace=True)

    if text_template == "fim_template_textual_encoding":
        # Convert text to tabular data
        for t in text:
            try:
                features_input = t.split(",")
                td = dict.fromkeys(columns, "NaN")

                # get data_name
                new_part_condition, new_part_values = t.split(" [sep] Input: ")
                input_textual_data = new_part_condition.removeprefix("Condition: data is ")

                input_values, output_part = new_part_values.split(" [sep] Target: ")
                # input_values = input_part.removeprefix("Input: ")
                output_values = output_part.removesuffix(" [answer]")

                features_input = input_values.split(",")
                features_output = output_values.split(" [answer] ")

            except ValueError:
                # If text is malformed, skip it
                continue

            if len(features_output) >= len(features_input):
                # Transform all features back to tabular data
                td["data"] = input_textual_data
                for i, f in enumerate(features_input):
                    col_i = f.strip().split(" is ")[0]
                    if col_i in columns:
                        try:
                            val_i = features_output[i]

                            if col_i in columns:
                                if feature_distribution[col_i][0] == "numerical":
                                    td[col_i] = float(val_i)  # Parse numerical values
                                else:
                                    td[col_i] = val_i  # Keep categorical as string
                        except ValueError:
                            dgt_logger.warning("Skipping malformed output: %s", features_output)
                            continue
            generated.append(td)
        df_gen = pd.DataFrame(generated)
        df_gen.replace("NaN", None, inplace=True)

    return df_gen


def fpc_transform_to_original_feature_space(
    generated_data,
    preprocessed_data,
    embedding_type,
    embedding_dims,
    data_embedded_full,
    data_embedded,
    fpc_basis,
    fpc_basis_full,
):
    n_var = len(preprocessed_data)  # Number of features

    # Reshape generated data to original format (for multivariate case)
    dgt_logger.info("Generated data shape is: %s", generated_data.shape)
    index = 0
    new_data_embedded = []
    for i in range(n_var):
        k_i = data_embedded[i].shape[1]  # Number of basis components
        new_data_embedded.append(generated_data[:, index : index + k_i])
        index += k_i
        dgt_logger.info("Reshaping for var_%s which uses %s basis components", n_var, k_i)

    dgt_logger.info(
        "Generated data shape has been reshaped to: %s", [arr.shape for arr in new_data_embedded]
    )

    # Retrieve to original form based on FPCA components
    new_data = []
    new_data_full = (
        [] if embedding_type == "fpc-filled" else None
    )  # Initialize for fpc-filled if required

    for var in range(n_var):
        # Retrieve the generated data back to the original feature space
        reconstructed_data = preprocessed_data[var].std(axis=0) * (
            new_data_embedded[var] @ fpc_basis[var].T
        ) + preprocessed_data[var].mean(axis=0)
        new_data.append(reconstructed_data)

        # For 'fpc-filled', add the generated data to the original data in the embedded space
        if embedding_type == "fpc-filled":
            sampled_indices = np.random.choice(
                data_embedded_full[var].shape[0],
                size=new_data_embedded[var].shape[0],
                replace=True,
            )
            data_embedded_full_var = data_embedded_full[var][sampled_indices]
            data_embedded_full_var[:, -embedding_dims[var] :] = new_data_embedded[var]
            reconstructed_full = preprocessed_data[var].std(axis=0) * (
                data_embedded_full_var @ fpc_basis_full[var].T
            ) + preprocessed_data[var].mean(axis=0)
            new_data_full.append(reconstructed_full)

    generated_data = new_data_full if embedding_type == "fpc-filled" else new_data

    return np.stack(generated_data, axis=0)


def fica_transform_to_original_feature_space(
    generated_data, preprocessed_data, data_embedded, fica_mixing, fica_mean
):
    n_var = len(preprocessed_data)  # Number of features

    # Reshape generated data to original format (for multivariate case)
    dgt_logger.info("Generated data shape is: %s", generated_data.shape)
    index = 0
    new_data_embedded = []
    for i in range(n_var):
        k_i = data_embedded[i].shape[1]  # Number of basis components
        new_data_embedded.append(generated_data[:, index : index + k_i])
        index += k_i
        dgt_logger.info("Reshaping for var_%s which uses %s basis components", n_var, k_i)

    dgt_logger.info(
        "Generated data shape has been reshaped to: %s", [arr.shape for arr in new_data_embedded]
    )

    # Retrieve to original form based on FICA components
    new_data = []
    for var in range(n_var):
        # Retrieve the generated data back to the original feature space
        reconstructed_data = np.dot(new_data_embedded[var], fica_mixing[var].T) + fica_mean[var]
        new_data.append(reconstructed_data)

    generated_data = new_data

    return np.stack(generated_data, axis=0)
