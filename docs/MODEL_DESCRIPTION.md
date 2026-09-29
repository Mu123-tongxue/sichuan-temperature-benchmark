# Implemented model hierarchy

## 1. Initial layered conductive model

`physics_vec()` implements the one-dimensional, steady-state, layered conductive calculation used as the physical baseline. Surface heat flow is spatially variable, while thermal conductivity and radiogenic heat production are piecewise constant by stratigraphic unit. The model produces the initial physics-based temperature `TPhys`.

## 2. Surrogate-assisted thermophysical calibration

RF, XGBoost, and MLP are used as **surrogates of the calibration objective**, not as direct formation-temperature predictors. `build_objective()` evaluates a candidate set of layer-wise thermal-conductivity and heat-production multipliers through the conductive forward model. `smbo_optimize()` performs the sequential surrogate-guided search and returns the best physically evaluated parameter combination.

The manuscript defaults are multiplier bounds 0.60–1.70, 60 initial candidates, 25 sequential iterations, 2000 surrogate-screened candidates per iteration, and 6 true forward-model evaluations per iteration.

## 3. Physics-informed neural network

`build_pinn_features()` combines spatial coordinates, terrestrial heat flow, burial depth, the initial conductive temperature, surface temperature, and stratigraphic-interface depths. `train_pinn()` implements a 256–256–128 fully connected network with `tanh` activations. Its loss combines observed-temperature fitting, conductive-prior consistency, a depth-monotonicity penalty, and a surface-boundary penalty with weights 1.0, 0.5, 0.2, and 0.3.

This implementation is physics-informed/physics-guided rather than a direct PDE-residual PINN: the conductive solution enters as a physical prior and the loss contains physically motivated constraints.

## 4. Hybrid PINN-XGBoost residual correction

`build_hybrid_features()` adds the PINN prediction to the physical and geological predictors. XGBoost is fitted to the residual `Tobs - TPINN`; the final prediction is `TPINN + residual_prediction`. The residual corrector therefore adjusts structured errors after the physics-informed main prediction and is not itself constrained by the conductive equation.

## 5. SHAP interpretation

`run_shap_analysis()` interprets the fitted XGBoost residual corrector. SHAP values describe predictor contributions to the **residual adjustment**, not causal controls of the complete geothermal field.
