# Experiment log

This is my detailed record of every experiment, in the order I ran them, including the ones that did not work. The README has the short version.
Paths use the current package layout (`src/windlidar/...`). Numbers come from my own runs on one CPU; retraining gives slightly different values.

## Synthetic-data reference results (first run, 4 000 cases, CPU)

Gust factor, `--features full`, test set (R2 / MAE): raw lidar 0.503 / 0.052; linear regression 0.833 / 0.027; best network (C, GELU)
0.837 / 0.026. Unseen set: raw 0.360 / 0.058; network 0.799 / 0.028. `--features level2`: network 0.810 / 0.028 (linear 0.810 / 0.028).
Uncertainty coverage of the nominal 95 % interval: 92-93 %. Turbulence intensity target (not used for the real test): network R2 0.958,
linear 0.956, raw lidar 0.659.
Reading: the correction roughly halves the error compared with the raw lidar, but a linear model does almost as well, because the remaining
error is mostly random (a single 10-min gust peak is a stochastic extreme). Your numbers will differ slightly. Use your own.

## Real-data results: FESSTVaL 2021, Falkenberg, ~90 m (my run, Nov 2026)

Data: sonic level 2 (`falkenberg_sonic90m`, `qc_wind==1`) and lidar level 2 (DWD product `qwind==1`, gate 92.7 m; UzK product, gate 90.3 m),
both on the same 10-min grid, periods with mean wind < 3 m/s removed. 9 238 (DWD) / 9 430 (UzK) matched periods, 18 May - 31 Aug 2021.
Code: `src/windlidar/fesstval/match.py` (matching), `src/windlidar/fesstval/analysis.py` (evaluation). Outputs: `results/fesstval/`.
Evaluation is a chronological hold-out (train < 15 Jul, test >= 15 Jul) plus 6-fold blocked CV. Target: sonic gust factor (3 s max / 10 min mean).

### 1. First attempt: the synthetic-trained network does not help
Using nearest-time matching and no quality flags (`fesstval/baseline_comparison.py`): raw lidar R2 = 0.26, corrected R2 = 0.22 (DWD); 0.06 / 0.07 (UzK).
After quality filtering and exact matching, on the hold-out (DWD): raw R2 0.277, RMSE 0.187; network 0.218, RMSE 0.195. The network is **worse than the raw lidar**.

### 2. Diagnosis
* The gust definitions agree (both about 3 s maxima), so it is not a definition bug. Lidar and sonic means agree (bias -0.01 m/s, r = 0.95).
* The emulator makes the lidar gust factor **too low** (smoothing), so the network adds a positive correction. The real lidar gust factor is already
  **too high by +0.04** (DWD; mean 1.35 vs 1.31), so the correction overshoots (bias +0.078).
* A noise scan of the emulator: with retrieval noise 0.05-0.30 m/s the emulated lidar GF is 0.05 *below* the sonic; with 0.3-0.7 m/s it is slightly above, as
  observed. A single 3.4 s circulation from ~10 beams is noisy, and the maximum of a noisy series is biased high.
* Not captured by the emulator at all: stability. Mean sonic GF is 1.44 by day (9-17 UTC) and 1.24 at night; the lidar overestimates by 0.08 by day and 0.015 at night.
  Real GF scatter (sd 0.16 of lidar minus sonic) is also larger than any noise level in the emulator reproduces (0.05-0.16).

### 3. Fix: calibrate the emulator noise (one parameter), retrain on synthetic data only
`gustnet/dataset.py --noise-range 0.3 0.7`, retrain (`data/dataset_noise_calibrated.csv`, `models/noise_calibrated/`). The noise range was chosen to match the real mean bias of
lidar minus sonic GF; for DWD that bias is identical in the training and test periods (0.038 / 0.038), so no test information leaks. No real data enters the network.

### 4. Results, hold-out (15 Jul - 31 Aug), DWD lidar (UzK in brackets)
| Model | bias | RMSE | R2 | blocked-CV R2 |
|---|---|---|---|---|
| raw lidar | 0.038 (0.043) | 0.187 (0.243) | 0.277 (-0.157) | 0.395 (0.030) |
| synthetic NN, original emulator | 0.078 (0.081) | 0.195 (0.241) | 0.218 (-0.142) | 0.305 (0.022) |
| **synthetic NN, noise-calibrated emulator** | 0.008 (0.001) | 0.176 (0.202) | 0.359 (0.197) | 0.489 (0.320) |
| linear recalibration on real data (GF) | -0.006 (0.004) | 0.174 (0.204) | 0.375 (0.178) | 0.541 (0.308) |
| linear + diurnal (stability proxy) | -0.006 (-0.002) | 0.173 (0.195) | 0.386 (0.251) | 0.537 (0.358) |
| gradient boosting + diurnal | -0.006 (-0.001) | 0.170 (0.186) | 0.405 (0.318) | 0.550 (0.394) |

### Reading, and limits
* Fixing one emulator parameter turns a harmful correction into a useful one: hold-out RMSE 0.187 -> 0.176 (DWD) and 0.243 -> 0.202 (UzK), and the bias is removed.
  It gets almost as far as a linear recalibration fitted on real data (0.174 / 0.204) without using any real target values in training.
* Real-data fits are only slightly better. Even the best model explains about 40 % (DWD) of the variance of the gust factor in the test period, because one 10-min
  gust peak is a stochastic extreme and the emulator is neutral-only. The diurnal feature helps clearly only for UzK.
* The noise calibration was done on DWD and transfers less well to UzK (UzK bias is 0.013 in the training period and 0.043 in the test period), and the noise
  range was chosen by hand from a coarse scan (5 settings), not optimised.
* One site, one summer, one height (lidar gate 92.7 m vs sonic 90.3 m for DWD). Block-CV fold-to-fold scatter in R2 is large (sd ~0.05).
* Not a validated lidar-turbulence correction. A stability-aware emulator (non-neutral spectra) would be the next step.

## Attempt: stability-aware emulator (does not close the gap)

Hypothesis: the synthetic-to-real gap comes from the neutral-only emulator (real gust factor is 1.44 by day, 1.24 at night) and from the sonic and lidar
seeing different air. Code: `src/windlidar/atmosphere/stability.py` (Monin-Obukhov: heat flux by hour -> zeta = z/L coupled to u*, stability-corrected profile, sigma_u/u*, length scale;
Davenport-type coherence between sonic and lidar volumes), `gustnet/dataset_stability.py`, `gustnet/calibrate.py` (random search + refinement of 5-6 parameters on
summary statistics of the **training period only**, < 15 Jul), `studies/stability_ablation.py` (train + ablations + real hold-out). Results: `results/stability_ablation.json`, `results/stability_emulator_params.json`.

What the calibrated emulator reproduces (training-period statistics, real -> simulated): mean gust factor day 1.446 -> 1.446, night 1.226 -> 1.264;
sonic GF spread 0.205 -> 0.203; lidar-sonic correlation 0.77 -> 0.85. What it does not: the lidar-minus-sonic bias at night is +0.07 in the emulator but +0.014 in the data,
i.e. the emulator has the stability dependence of the lidar bias the wrong way round. Calibrated parameters sit at or near their bounds (H_day 0.50 K m/s is on the high side
of physical; sig_spread 0.39 at its bound), so some of the fit is compensating for missing physics.

Hold-out (15 Jul - 31 Aug), network trained on the emulator, DWD (UzK in brackets); R2 / RMSE:
| Training emulator | R2 | RMSE |
|---|---|---|
| neutral + constant noise (earlier fix) | 0.359 (0.197) | 0.176 (0.202) |
| stability + decoherence, features U, GF | 0.345 (0.082) | 0.178 (0.216) |
| stability + decoherence, features U, GF, hour | 0.300 (0.069) | 0.184 (0.218) |
| decoherence only (no stability), U, GF, hour | 0.356 (0.150) | 0.177 (0.208) |
| stability, no decoherence | 0.312 (0.019) | 0.183 (0.223) |
| for reference: linear fit on real data | 0.375 (0.178) | 0.174 (0.204) |

Findings: (1) adding the stability physics did **not** improve the hold-out result; the neutral emulator with decoherence is as good or better. (2) Decoherence between the
two instruments is the only ingredient that consistently helps (R2 +0.04 to +0.05 in the ablations). (3) Hour-of-day features do not help the network, because the emulator's hour -> stability
mapping is an assumption and its lidar-bias pattern disagrees with the data. (4) Making the lidar retrieval noise proportional to sigma_u (turbulence inside the scan cone) did not fix the night-time bias
(fitted fraction 0.16, loss unchanged). Conclusion: the gap is not mainly caused by stability, so the remaining error is probably in how the real lidar derives its gust
(scan geometry, filtering, w-component) or in sensor separation, none of which this emulator models. A fix would need the level-1 lidar data (tens of GB) and a beam-level simulation.

## Reframing: lidar signal processing, wind-field generation and design wind conditions (`src/windlidar/`)

Motivation: the lidar gust-factor correction (above) was limited by how the lidar derives its wind and by instrument separation, not by the neural network.
The `src/windlidar/` package works on those causes directly with a spatially coherent virtual lidar, then uses the result for design-condition and generative-model questions.
Everything below is synthetic or uses the FESSTVaL level-2 data already in `real_data/`; no new downloads.

| Module | What it does |
|---|---|
| `lidar/virtual.py` | 3-D coherent turbulence at the beam sample points (Veers method, IEC coherence, Taylor), FESSTVaL CSM geometry (62 deg, 10 beams / 3.4 s, 30 m gate), least-squares VAD retrieval, radial-velocity noise |
| `studies/signal_processing.py` | spectral transfer function of the lidar, blind noise-floor estimate, turbulence-intensity restoration (train/test by record) |
| `studies/noise_sweep.py` | retrieval noise needed to reproduce the real lidar-minus-sonic gust-factor bias |
| `design/iec.py` | IEC 61400-1 NTM / ETM / EOG / EWM and a check against the FESSTVaL measurements |
| `generation/ddpm.py` | conditional diffusion model for 64 s turbulent wind series with optional IEC-EOG gust, compared with Kaimal synthesis |

### Signal-processing results (virtual lidar, 800 records, `results/analysis/lidar_signal_processing.json`)
* The scan cone behaves like a boxcar of only about 25-35 m (best fit per wind-speed bin), not the 40-150 m range assumed in `src/windlidar/lidar/gust_proxy.py`. Most of the lost variance comes from
  the 3.4 s sampling (aliasing), not from spatial smoothing: the lidar keeps only about 55-57 % of the sonic variance, nearly independent of wind speed.
* The blind noise-floor estimator (median of the spectrum above 0.115 Hz) **fails**: it overestimates the noise variance by a median factor 2.5 (10-90 %: 1.2-8.7) because aliased turbulence fills that band.
  Subtracting it makes the turbulence intensity worse (R2 0.09, raw 0.57).
* A data-driven attenuation ratio R(U), fitted on half of the records, restores TI on the other half: RMSE 0.0236 -> 0.0139, R2 0.57 -> 0.85, bias -0.020 -> -0.002.
  This works only because the fit absorbs the noise-estimator bias; it is not a physical noise separation.
* Gust factor: the virtual lidar reads **lower** than the sonic (bias -0.04, correlation 0.83) at radial noise 0.1-0.4 m/s, the opposite of the real DWD lidar (+0.04).
  The sweep shows the sign flips at an effective speed-noise of about 0.45 m/s and matches +0.04 at about 0.65 m/s, in line with the 0.4-0.7 m/s found by the emulator calibration.
  That is a consistency check, not a proof that the real noise is that high (other error sources look the same).

### Design wind conditions (`results/analysis/iec_design_check.json`)
* NTM check: with the attenuation correction above (variance ratio 0.56, assumption), the 90th-percentile sigma_u at ~90 m stays below the class-C NTM curve in all wind-speed bins
  (e.g. 1.06 vs 1.29 m/s at 6-8 m/s). This is summer 2021 at one site, so it is an indication, not a site classification. The raw lidar sigma is lower still.
* EOG check: 45 % of the 10-min sonic records have a 3 s gust excess above the 3 s-smoothed EOG rise of a class IIB turbine. This is **not** an IEC exceedance statistic: the EOG is defined for a rotor-averaged
  gust (D = 126 m, factor 1/(1+0.1 D/Lambda1)), the sonic measures at a point. The comparison shows why a lidar-derived rotor-effective wind speed is needed for design-condition checks.

### Wind-field generation: conditional diffusion model (`src/windlidar/ddpm_wind.py`, `results/analysis/diffusion_eval.json`)
0.5 M-parameter 1-D U-Net DDPM (200 steps, 4 000 training steps on CPU, about 17 min), 64 s series at 10 Hz in units of sigma_u, conditioned on L_u/U, a gust flag and the gust amplitude
A = Vgust/sigma (IEC EOG shape at a random time). Trained on 13 500 Kaimal windows (40 % with EOG). Evaluated on 256 samples per condition against Kaimal synthesis plus the analytic EOG (the baseline is correct by construction):

| Condition | PSD ratio gen/ref (0.05-0.2 / 0.2-1 / 1-5 Hz) | std gen / ref | 1 s increment kurtosis gen / ref | recovered gust A (requested) |
|---|---|---|---|---|
| normal | 0.93 / 0.94 / 0.87 | 0.60 / 0.63 | 3.12 / 3.15 | - |
| EOG, A = 3 | 0.85 / 0.93 / 0.86 | 0.70 / 0.74 | 5.17 / 5.16 | 2.74 +- 0.67 (ref 3.03 +- 0.45) |
| EOG, A = 5 | 0.83 / 0.93 / 0.87 | 0.84 / 0.90 | 7.95 / 8.12 | 4.38 +- 0.95 (ref 4.93 +- 0.46) |

Reading: the model reproduces the spectral shape within 6-17 %, the Gaussian increment statistics and the heavy-tailed increments caused by the gust, and it responds to the requested amplitude.
It is **not as accurate as the classical baseline**: variance and power are 6-14 % low (largest at low frequency for the gust cases), and the gust amplitude is about 10 % too small with roughly twice the scatter.
That is expected, because the training data are Gaussian Kaimal plus an analytic gust, so there is nothing to learn that the analytic model does not already give exactly. The value of such a generator would be
training it on measured high-rate lidar/sonic data where no analytic model exists (non-Gaussian, intermittent, stability-dependent turbulence). That has not been done here.
Not tested: spatial fields (only one point in time), the IEC coherence between points, more than one training seed.

### How the CFD case connects to the reframed project (`src/windlidar/cfd_link.py`, `results/analysis/cfd_*.{json,csv}`)
The OpenFOAM case is used in two ways: as the mean-flow/closure model whose validity is tested against real lidar profiles, and as the input that sets turbulence level and terrain roughness for the virtual lidar and the IEC check.

**What the CFD does and does not provide.** In all 16 sweep runs k at 90 m equals u*^2/sqrt(Cmu) within 1 % (Richards-Hoxey boundary conditions fix it), so the RANS turbulence is set by u* alone and does not depend on z0.
RANS sigma_u = sqrt(k/0.945) = 1.88 u*, against 2.29 u* for neutral surface-layer turbulence (Panofsky-Dutton limit; 2.4 typical). The closure is therefore a factor 1.22 low in sigma_u, which is applied as a correction below.

**A. Validation against FESSTVaL lidar profiles (DWD level 2, Falkenberg).** Median 10-min wind profiles for three regimes; u*, z0 fitted to the 66-146 m gates (the 13 and 40 m gates have no valid data, so the surface layer cannot be checked);
OpenFOAM run with those values; compared with the measured profile. Agreement in the fit region (4 gates, 2 parameters) is by construction. The independent test is 150-437 m:

| Regime | n profiles | u*, z0 (fit) | CFD vs lidar, 150-437 m: bias / RMS |
|---|---|---|---|
| high wind, U92 9-13 m/s | 451 | 0.62 m/s, 0.13 m | -0.05 / 0.12 m/s |
| daytime, 10-15 UTC, U 4-8 m/s | 1 401 | 0.28 m/s, 0.025 m | +0.11 / 0.17 m/s |
| night, 22-02 UTC, U 4-8 m/s | 1 876 | 1.20 m/s, **z0 = 11 m** | not applicable |

The neutral RANS profile follows the measured profile to about 1-2 % of the wind speed up to 437 m for high-wind and daytime conditions. At night the log-law fit returns an unphysical roughness (z0 about 4-12 m between 19 and 05 UTC,
against 0.005-0.06 m during the day), the signature of stable stratification: **a neutral RANS closure is not valid for the stable nights at this site**. (An earlier version of this comparison clipped z0 to fit the mesh and reported an 11.8 m/s CFD error for the night case; that was an artefact of the clipping, not a CFD error, and is not reported.)
Limits: medians, not individual profiles; high wind is assumed near-neutral (no independent stability measurement); only 4 fit gates; summer 2021 only.

**B. CFD-driven design and lidar check (16 runs, u* 0.30-0.75 m/s, z0 0.01-0.20 m, `cfd_design_table.csv`).** With sigma_u corrected to 2.29 u* and U from the RANS at 90 m, the implied mean-state IEC reference intensity
Iref = sigma_u / (0.75 U + 5.6) is 0.065-0.123: below class C (0.12) in 15 of 16 cases, reaching class C only for the roughest, fastest case (u* = 0.75, z0 = 0.2 m). It rises with both u* and z0.
This agrees with the real-data NTM check above (site turbulence below class C), but note that IEC's sigma1 is a 90th-percentile value, so the mean-state Iref understates it by roughly a quarter.
Using the uncorrected RANS sigma_u would give a turbulence intensity 18 % too low. Running the virtual lidar at these conditions (12 realisations per case, radial noise 0.2 m/s): raw lidar TI is on average 17 % below the true TI;
with the signal-processing correction from `studies/signal_processing.py` (blind noise subtraction + R(U)) it is within +3 % (mean error +0.003, RMS 0.009 in TI).
