# Ocean Acoustics Analysis - PDSG Feb 2020

We're analysis acoustic data from the Aloha Cabled Observatory, 
with a subset of 5 min chunks from every other hour over a year (2015)
or a month (Feb 2015).

1) Load up files.

2) Create spectrogram

3) Plot waveform.

4) Analyse FFTs.  Build up power spectrum for each file, average together.

Look for daily/seasonal differences.

5) Build up anomaly detection algorithm.
Based on subtracting off likely background, try to identify anomalies. 

Could try to use a neural network for fast detection of anomalies. 
Variational autoencoders? 
