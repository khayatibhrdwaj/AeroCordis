# Cardiopulmonary Digital Twin: System Enhancement Design

## 1. Introduction

This document outlines the proposed enhancements to the existing cardiopulmonary digital twin system, focusing on integrating advanced predictive capabilities using Long Short-Term Memory (LSTM) networks, implementing robust n:m phase locking for cardiorespiratory synchronization, and upgrading the user interface to a professional, clinical-grade dashboard. The goal is to leverage the existing modular architecture to deliver a sophisticated tool for studying ECG and respiratory system interactions using MIMIC-IV data.

## 2. LSTM Prediction Integration

The current `TrendPredictor` in `src/prediction/trend_predictor.py` includes an `CardiopulmonaryLSTM` model definition and online training logic. However, the `predict` method currently only returns linear slopes and Estimated Time to Alert (ETA) values, without incorporating the LSTM's predictive output. This section details how to fully integrate the LSTM predictions.

### 2.1. LSTM Prediction Generation

1.  **Prediction Horizon**: The LSTM model is designed to predict `horizon` steps into the future. The `predict` method will call the LSTM model (if enabled and `HAS_TORCH` is true) to generate future sequences for HR, RR, SpO2, O2 delivery, HRV, and Stress load.
2.  **Data Normalization**: The LSTM model expects normalized input. The `_online_train` method already handles normalization for training. A similar normalization scheme (using the mean and standard deviation of the `_feature_buf`) will be applied to the input sequence before feeding it to the `_model.eval()` for prediction.
3.  **Denormalization**: The LSTM output will be denormalized back to the original scale using the same mean and standard deviation used for normalization.

### 2.2. Blending Predictions (Combined Mode)

If `cfg.prediction.method` is set to `combined`, the system will blend the linear extrapolation with the LSTM predictions. A configurable weighting factor (`cfg.prediction.combined_weight_lstm`) will determine the influence of the LSTM output.

$$Prediction_{final} = (1 - w) \times Prediction_{linear} + w \times Prediction_{LSTM}$$

Where $w$ is `cfg.prediction.combined_weight_lstm`.

### 2.3. ETA Calculation with LSTM Predictions

The `_eta_to_threshold` and `_eta_to_threshold_two_sided` methods currently rely on linear slopes. These methods will be updated to use the predicted future values from the LSTM (or combined prediction) to estimate the time to threshold crossing. This will involve iterating through the predicted future sequence to find the first point where a threshold is crossed.

### 2.4. Online Training Refinement

The `_online_train` method will be reviewed to ensure efficient and stable online learning. Considerations include:

*   **Training Frequency**: The current training every 10 steps might be too frequent or infrequent depending on data characteristics. This will be made configurable.
*   **Loss Function**: `nn.MSELoss()` is appropriate for regression. No changes planned unless performance dictates otherwise.
*   **Data Window**: Ensure the `_feature_buf` provides sufficient and relevant history for training.

## 3. N:M Locking for Phase Synchronization

The `TwinState` dataclass includes `n_ratio` within `CouplingFeatures`, but the `PhaseSynchronisation` module in `src/coupling/phase_sync.py` does not currently compute this value. This enhancement will implement the calculation of the n:m locking ratio.

### 3.1. Theoretical Background

Cardiorespiratory phase synchronization often exhibits n:m locking, where n cardiac cycles correspond to m respiratory cycles. Detecting this requires analyzing the instantaneous frequencies or phase velocities of both signals and identifying stable integer ratios between them over a certain period.

### 3.2. Implementation in `PhaseSynchronisation`

1.  **Instantaneous Frequencies**: After computing the instantaneous phases of ECG and respiratory signals using the Hilbert transform, their instantaneous frequencies will be derived by taking the time derivative of the unwrapped phases.
2.  **Ratio Calculation**: A sliding window approach will be used to calculate the ratio of the average instantaneous heart rate frequency to the average instantaneous respiratory rate frequency within that window.
3.  **Stability Check**: To identify stable n:m locking, the calculated ratio will be compared against integer or simple fractional values (e.g., 2:1, 3:1, 4:1, 1:1, 3:2, 5:2). A tolerance will be applied to account for biological variability. The `n_ratio` will store the most prominent stable ratio found.
4.  **Configuration**: New configuration parameters will be added to `TwinConfig` for `phase_sync` to control the window size for ratio calculation and the tolerance for identifying integer ratios.

## 4. Professional Dashboard and Visualization

The existing `src/visualization/dashboard.py` provides a basic Streamlit interface. To achieve a 
professional, clinical-grade interface, the following improvements are proposed:

### 4.1. Enhanced Real-time Data Display

*   **High-Fidelity Waveform Plots**: Improve the rendering of ECG, respiratory, and SpO2 waveforms with options for zooming, panning, and event markers (e.g., R-peaks, breath cycles).
*   **Dynamic Metric Cards**: Display key physiological metrics (HR, RR, SpO2, HRV, O2 Delivery, Stability, Stress Load) with clear labels, units, and color-coded status indicators (normal, warning, critical).
*   **Trend Visualizations**: Implement interactive trend plots for all tracked metrics, allowing users to select time windows (e.g., 1 hour, 6 hours, 24 hours) and overlay alert thresholds and predicted trajectories.

### 4.2. Advanced Coupling Visualizations

*   **RSA Visualization**: Plot heart rate variability alongside respiratory cycles to visually represent RSA. Consider a spectrogram or similar frequency-domain plot to show coherence.
*   **Cross-correlation Plot**: Display the full cross-correlation series (`xcorr_series`) between HR and RR, highlighting the peak and lag.
*   **Phase Synchrony Plot**: Visualize phase relationships between ECG and respiratory signals. This could involve Poincaré plots of phase differences or a continuous plot of instantaneous phases, clearly indicating periods of n:m locking.
*   **Hypoxia Response Summary**: A dedicated panel to summarize hypoxia events, including SpO2 drops, corresponding HR changes, and the blunted response flag.

### 4.3. Alert Management and Prediction Interface

*   **Alert Log**: A clear, sortable, and filterable log of all `AlertEvent` objects, showing timestamp, severity, channel, message, value, and threshold.
*   **Predictive Timeline**: A visual timeline indicating predicted threshold crossings (ETAs) for critical parameters, showing the parameter, predicted time, and severity.
*   **Configuration Access**: Allow clinicians to adjust alert thresholds and prediction horizons directly from the dashboard (with appropriate authentication and saving mechanisms).

### 4.4. MIMIC-IV Data Selection and Playback

*   **Subject/Stay Selector**: A user-friendly interface to browse available MIMIC-IV subjects and their ICU stays, allowing selection of specific records for analysis.
*   **Playback Controls**: Implement controls for playing, pausing, fast-forwarding, and rewinding the waveform data stream, enabling retrospective analysis of clinical events.
*   **Data Export**: Functionality to export processed features, trends, and alerts for selected time periods into common formats (CSV, JSON).

### 4.5. Technology Stack

*   **Frontend**: Streamlit will be retained for rapid prototyping and ease of integration with Python backend. However, custom HTML/CSS/JavaScript components will be used for more advanced visualizations where Streamlit's native widgets are insufficient.
*   **Plotting**: Plotly will be the primary library for interactive plots, leveraging its capabilities for time-series data and scientific visualizations.

## 5. MIMIC-IV Data Pipeline Refinement

The `MIMICLoader` in `src/data/mimic_loader.py` provides basic streaming functionality. Refinements will focus on robustness and ease of use.

### 5.1. Error Handling and Robustness

*   **Missing Channels**: Improve handling of records where expected channels (ECG, Resp, SpO2) are missing or have poor quality. Provide clear feedback to the user.
*   **Data Gaps**: Implement strategies for handling gaps in waveform data, such as interpolation or flagging periods of missing data.
*   **Authentication**: Ensure secure and persistent authentication with PhysioNet, potentially integrating with environment variables or a secure configuration store.

### 5.2. Metadata Integration

*   **Clinical Context**: Integrate relevant clinical metadata from MIMIC-IV (e.g., patient demographics, diagnoses, medications, lab results) into the `TwinState` or a separate context object. This will enrich the digital twin with clinical context for better interpretation of physiological changes.
*   **Event Markers**: Overlay clinical events (e.g., medication administration, ventilator changes) onto waveform and trend plots to correlate physiological responses with interventions.

### 5.3. Efficient Data Access

*   **Pre-processing/Caching**: For frequently accessed records or for performance, consider pre-processing and caching waveform data locally after initial download.
*   **Record Selection Logic**: Enhance the logic for selecting the most appropriate waveform record for a given `stay_id`, especially when multiple records exist.

## 6. Development Roadmap and Timeline

Given the ambitious scope and the user's desire to complete this by today (6-7 hours of work), a realistic approach involves prioritizing core functionality and delivering a strong foundation with clear next steps. The following is a proposed roadmap:

### Phase 1: Core LSTM Integration (2-3 hours)

*   Implement LSTM prediction generation and denormalization.
*   Integrate LSTM predictions into ETA calculations.
*   Refine online training parameters.

### Phase 2: N:M Locking Implementation (1-2 hours)

*   Implement instantaneous frequency derivation.
*   Develop n:m ratio calculation and stability check within `PhaseSynchronisation`.
*   Add configuration parameters for n:m locking.

### Phase 3: Dashboard Enhancements (2-3 hours)

*   Focus on key visualizations: enhanced waveform plots, dynamic metric cards, and interactive trend plots.
*   Implement basic alert log and predictive timeline.
*   Prioritize a clean, professional aesthetic.

### Phase 4: MIMIC-IV Data Refinement & Documentation (Remaining time)

*   Improve error handling for missing channels and data gaps in `MIMICLoader`.
*   Add basic clinical metadata integration (e.g., displaying patient age/gender).
*   Prepare comprehensive documentation, including setup instructions, usage guide, and a detailed explanation of the implemented features and future work.

This phased approach ensures that a functional and impressive core system is delivered within the tight timeframe, with clear pathways for further development. The 
