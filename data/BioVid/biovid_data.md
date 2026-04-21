The BioVid Heat Pain Database (Part A) (Walter et al., 2013)
is a multi-modal data set consisting of 87 healthy participants
subjected to four levels of gradually increasing and individually
calibrated thermal pain elicitation (T1, T2, T3, T4). Several
modalities were recorded during the experiments including
video streams, EDA, ECG, and EMG signals. Each single
level of pain elicitation was randomly elicited a total of 20
times, with each elicitation lasting 4 s (sec), followed by a
recovery phase of randomized duration (lasting between 8 and
12 s). During this recovery phase, a baseline temperature T0
of 32◦C was applied (see Figure 5). The data set specific to
each participant consists of a total of 20 × 5 = 100 samples,
summing up to a database of 87 × 100 = 8, 700 samples.
Each sample is labeled with its corresponding level of thermal
pain elicitation (T0, T1, T2, T3, T4). The proposed approaches
are evaluated uniquely on the physiological signals EMG,
ECG, and EDA.

## Preprocessing
First of all, the sampling
rate of the recorded signals was reduced to 256 Hz in order to
significantly reduce the amount of computational requirements.
Next, the amount of noise and artifacts within each signal was
reduced by applying specific signal processing techniques. For
both datasets, a low-pass Butterworth filter of order 3 with a
cut-off frequency of 0.2 Hz was applied on the EDA signals.
Concerning the BioVid Heat Pain Database, EMG signals were
filtered using a fourth order bandpass Butterworth filter with a
frequency range of [20, 250] Hz, while ECG signals were filtered
with a third order bandpass Butterworth filter with a frequency
range of [0.1, 250] Hz. Subsequently, piecewise detrending of the
filtered ECG signals was performed, by subtracting a fifth degree
polynomial least-squares fit fromthe filtered signals (as proposed
in Thiam et al., 2019a). Concerning the SenseEmotion Database,
the RSP signals were smoothed using a third order low-pass
Butterworth filter with a cut-off frequency of 0.8 Hz. Both EMG
and ECG signals were preprocessed by applying a third order
bandpass Butterworth filter with respective frequency ranges
of [0.05, 25] and [0.1, 25] Hz, followed by a similar piecewise
detrending as in the case of the BioVid Heat Pain Database.
The resulting filtered signals were subsequently segmented, and
each segment in combination with its corresponding level of
pain elicitation was used to perform the assessment of the
proposed approaches.
In the case of the BioVid Heat Pain Database, the assessment
is performed on windows of length 4.5 s with a shift of 4 s from
the elicitations’ onset (see Figure 6).