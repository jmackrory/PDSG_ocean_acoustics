# Ocean Acoustics Analysis - PDSG Feb 2020

We're analysis acoustic data from the ALOHA Cabled Observatory, 
with a subset of 5 min chunks from every other hour over a year (2015)
or a month (Feb 2015).

The data was collected and preprocessed by Philip Robinson.


1) Load up files.

2) Create spectrogram

3) Plot waveform.

4) Analyse FFTs.  Build up power spectrum for each file, average together.

Look for daily/seasonal differences.

5) Build up anomaly detection algorithm.
Based on subtracting off likely background, try to identify anomalies. 

Could try to use a neural network for fast detection of anomalies. 
Variational autoencoders? 


"Frequencies of interest from Listening for Whales at the Station ALOHA Cabled Observatory"
by J.N Oswald in Listening in the Ocean.

Blue whales: fundamental around 15-20Hz for 10-20 sec
Fin Whales: 20 Hz.  FM data from 30-40 Hz down to 20Hz over a second.
Sei whales: 3 kHz.  Also 100Hz down to 3 Hz over a second. 
Minke around 1.4 kHz "boing"

Apparently, June through October is the quiet season (Whales migrate away back to cooler waters)
