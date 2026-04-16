# Joint Literature Review — Updated Version
# Changes: Paragraphs 5–6 (Artem's surrogate section) rewritten with Galliamov et al. 2026 citation
# and new references not present in the original paper.
# All other paragraphs kept verbatim from Gleb's literature_review.docx.
#
# NEW REFERENCES added (not in Galliamov et al. 2026 original paper):
#   — Raissi et al. 2019 (PINNs) — contrast: physics-constrained vs. physics-emulating surrogates
#   — Bhosekar & Ierapetritou 2018 (surrogate advances review) — broader surrogate context
#   — Razavi et al. 2012 (surrogate modeling review in engineering) — foundational surrogate review
#   — Gholami et al. 2019 (smart proxy CO2-EOR) — already in Gleb's version, kept
# ─────────────────────────────────────────────────────────────────────────────

## 2. Literature Review

Hydraulic-fracture flowback is a short but operationally critical stage that links stimulation design
to long-term well productivity. In unconventional reservoirs the process is governed by coupled
multiphase flow, progressive fracture-conductivity degradation, geomechanical closure, and
near-fracture fluid redistribution, all of which make high-fidelity simulation physically informative
yet computationally expensive (Osiptsov et al., 2020; Boronin et al., 2023). For tight-oil formations
such as the Bazhenov, a mechanistic flowback framework has been progressively developed to
incorporate fracture cleanup dynamics, gel-rheology effects, proppant embedment, and
startup-schedule optimization. Crucially, this modeling line is not purely theoretical:
field-testing campaigns and subsequent calibration studies have demonstrated that the framework can
be matched to real multistage-fractured horizontal wells in Western Siberia, confirming its status
as a credible physical basis rather than a synthetic benchmark alone (Vainshtein et al., 2020, 2021;
Boronin et al., 2023). The unresolved problem, therefore, is no longer whether physically grounded
flowback models can describe field behavior, but how such models can be made computationally
tractable for repeated calibration, sensitivity analysis, and multi-well optimization.

One established answer to this computational bottleneck is surrogate modeling. In petroleum
engineering, surrogate (proxy) models are now widely used to emulate expensive numerical simulators
for screening, uncertainty quantification, and design optimization (Razavi et al., 2012;
Bhosekar and Ierapetritou, 2018). Representative petroleum applications include neural-network
surrogates for hydraulic-fracture propagation (Kalyuzhnyuk et al., 2019), artificial-neural-network
proxies for advanced well-completion design under uncertainty (Moradi et al., 2022), smart proxy
models for CO₂-EOR performance forecasting (Gholami et al., 2019), data-driven workflows for
hydraulic-fracture design and production prediction (Morozov et al., 2020), and machine-learning
pipelines for production forecast and parent–child well optimization in unconventional settings
(Wang et al., 2021). A similar strategy appears in offset-well design, where a surrogate is
coupled to a metaheuristic optimizer to tune completion geometry in the Bakken formation (Merzoug
and Rasouli, 2023). Collectively, these studies confirm that forward surrogate modeling in the
petroleum domain is already a mature technological layer: once trained, the proxy replaces a costly
simulator with a fast emulator capable of supporting repeated scenario evaluation at negligible
marginal cost.

However, the same body of work reveals a systematic limitation. The overwhelming majority of
petroleum surrogate studies remain fundamentally forward-only: the proxy is used to reproduce
simulator outputs or to accelerate design-space exploration, but the loop is not explicitly closed
by performing inverse calibration of physical model parameters against field observations. This
distinction matters. A surrogate used for forward prediction answers the question "what output
follows from a given parameter set?", whereas a surrogate embedded in an inverse workflow addresses
the substantially harder question "which physically admissible parameter set explains the observed
field response?". In the fracture and completion literature, even the closest prior works stop short
of this second step. Kalyuzhnyuk et al. (2019) demonstrated a neural surrogate for fracture growth
but restricted attention to forward emulation; Moradi et al. (2022) addressed completion design under
uncertainty without field-level inverse calibration; Gholami et al. (2019) developed a smart proxy
for reservoir performance forecasting; and Merzoug and Rasouli (2023) coupled a surrogate with
optimization, yet the optimization target was well-design parameters rather than calibration of
physical flowback quantities to production data. Thus, existing solutions cover forward acceleration
convincingly but do not provide an end-to-end forward–inverse architecture for fracture flowback.

This gap is consistent with the broader flowback literature. The recent review by Lv et al. (2025)
shows that proppant-flowback research remains fragmented across experimental studies, control
methods, and predictive approaches and still lacks broadly applicable predictive frameworks
grounded in integrated physics. In other words, the community recognizes the importance of flowback
control and prediction but does not yet offer a comprehensive architecture that combines a
field-validated physical model, a computationally efficient surrogate layer, and an explicit
inverse-calibration stage. This observation is central to positioning the present work: the novelty
does not lie in proposing yet another proxy model or in restating that history matching is useful,
but in coupling these elements into a single pipeline for a flowback problem where such coupling
has not been convincingly demonstrated.

### [UPDATED — replaces paragraphs 5–6 from Gleb's original] ###

A second important distinction concerns the surrogate formulation itself. Two broad strategies
exist in the literature for embedding physical knowledge into machine-learning models. The first —
physics-informed neural networks (PINNs) — incorporates governing equations directly into the
training loss as soft constraints (Raissi et al., 2019), which allows the network to satisfy
conservation laws approximately even in regions with sparse training data. The second strategy,
which we adopt, is physics-emulating: the surrogate is trained on a large synthetic dataset
generated by a validated high-fidelity simulator and is required to reproduce simulator outputs
rather than to satisfy PDEs during training. The physics-emulating approach is preferable when the
simulator already encodes the relevant physics at high fidelity and when the primary goal is
computational acceleration rather than extrapolation beyond the training domain. For the fracture
flowback problem, the mechanistic model of Boronin et al. (2023) provides precisely this validated
foundation.

The surrogate developed in our prior work (Galliamov et al., 2026) implements this physics-emulating
strategy with one structurally important addition: the simulator's input space is first compressed
through classical dimensional analysis before being passed to the machine-learning layer. The
original flowback problem involves more than fifty physical parameters; through Buckingham Pi
dimensional analysis, these are reduced to eight governing dimensionless groups that encode fracture
geometry, gel rheology, reservoir-to-fracture permeability contrast, proppant embedment thresholds,
and compaction characteristics. A multilayer perceptron (MLP) trained on ten thousand synthetic
simulations then maps these eight groups to three target flowback quantities — effective cleaned
fracture length, mean post-flowback permeability, and mean aperture — achieving a mean absolute
percentage error of approximately five percent on held-out data under Newtonian-rheology conditions,
confirmed by five-fold cross-validation. This dimensionless interface is not merely a convenience
for machine learning: it imposes dynamic similarity across different wells and operating conditions,
making the surrogate physically portable rather than tied to a single reservoir. From the standpoint
of downstream inverse calibration, the dimensionless formulation also sharply reduces the effective
search space, which is one of the key reasons a surrogate-based inversion in this representation is
preferable to direct inversion in the full physical parameter space — a point underscored by the
broader surrogate literature (Bhosekar and Ierapetritou, 2018; Razavi et al., 2012).

Interpretability of the surrogate is not an auxiliary feature in this context; it is a prerequisite
for credible inverse use. Sensitivity analyses reported in Galliamov et al. (2026) using SHAP values
and variance-based Sobol decomposition confirm that the trained MLP recovers physically consistent
feature rankings. In non-Newtonian regimes the dimensionless parameter associated with yield-stress
effects and the critical pressure gradient dominates the surrogate response, which is consistent with
the physical expectation that unbroken gel impedes fracture cleanup. In Newtonian regimes the
permeability ratio and fracture-geometry parameters become the leading factors. Furthermore, large
gaps between first-order and total-order Sobol indices indicate that the system is governed by
strong higher-order parameter interactions rather than by additive single-parameter effects. These
findings provide confidence that the surrogate preserves the sensitivity structure of the underlying
mechanistic model and can therefore be embedded in an inverse workflow without collapsing the
problem into a purely black-box optimization. The present paper builds directly on this established
surrogate foundation, extending it into a coupled forward–inverse pipeline and demonstrating
end-to-end calibration against field production data.

### [END OF UPDATED SECTION — Gleb's text continues unchanged below] ###

The choice of inverse methodology follows directly from the structure of the flowback problem.
History matching in this setting is strongly ill-posed unless parameter selection is guided by
sensitivity analysis and the calibration task is decomposed into physically motivated stages. In the
wells considered here, sensitivity analysis identified matrix permeability and, where applicable,
fracture half-length as the dominant tuning variables, whereas parameters such as fracture aperture
and porosity proved much less influential over the flowback interval. The adaptation algorithm
itself is an automated history-matching loop based on finite-difference gradient descent with
parameter-specific step scaling and a backtracking line search (Armijo criterion). The calibration
is explicitly decomposed into two layers: the initial water-saturation profile in the near-fracture
zone is adjusted separately from the global transport parameters because it controls water-cut
dynamics, while permeability and fracture half-length primarily govern the oil-production response.
This decomposition is not a cosmetic algorithmic choice; it is a practical remedy for equifinality
in a multiparameter inverse problem where simultaneous adjustment of all unknowns leads to
non-unique and physically inconsistent solutions.

An especially instructive result from the adaptation study underscores why numerical fit alone is
not a sufficient criterion for model adequacy. In one well, a simplified model achieved a prediction
error comparable to that of the full model, but only by inferring a substantially shorter fracture
half-length than indicated by fracture-design data; the calibrated geometry was therefore physically
inconsistent despite a satisfactory error metric. In another well, the simplified model performed
markedly worse, with cumulative-production mismatch rising to approximately twenty-one percent,
because transient fracture-conductivity effects were not adequately represented. These results are
directly relevant to the rationale for the proposed pipeline. They demonstrate that the purpose of
inverse calibration is not simply to minimize a mismatch metric but to recover a parameter set that
remains physically interpretable. A forward surrogate that is later used inside an inverse loop must
therefore preserve physically meaningful sensitivity structure, not merely deliver low regression
error.

A brief cross-domain analogy reinforces the methodological generality of the proposed architecture
without overstating it. In numerical weather prediction, recent machine-learning models such as
GraphCast and Pangu-Weather have demonstrated that learned surrogates can deliver fast and skillful
forward forecasts once trained on large reanalysis datasets (Lam et al., 2023; Bi et al., 2023).
Structurally, this resembles the petroleum problem: an expensive physics-based forecasting system
is complemented by a learned forward model that dramatically reduces runtime. Yet these weather
models remain primarily forward predictors; they do not, by themselves, provide a physics-grounded
inverse-calibration loop analogous to parameter adaptation against flowback observations. The
analogy is therefore architectural rather than evidential. It supports the claim that the sequence —
validated physical model, dimensionally reduced representation, fast surrogate, inverse adaptation —
is not domain-specific, while fracture flowback remains the primary demonstration domain of the
present study.

At the same time, the critical literature identifies several unresolved limitations that apply to
any surrogate-assisted inverse workflow. First, surrogate approximation error is routinely neglected
during inversion, even though it can bias estimated parameters and produce overconfident
conclusions. Second, while dimensionless parameterization is increasingly recognized as beneficial
for surrogate robustness, its quantitative impact inside a full calibration loop has not been
systematically isolated from other sources of improvement. Third, decomposed calibration strategies
— in which initial conditions and global transport parameters are adapted separately — appear
practically promising but remain theoretically underexplored; formal convergence guarantees and
sensitivity to decomposition order are largely absent from the current literature. Fourth, formal
uncertainty quantification is not yet integrated into the pipeline: confidence intervals on
calibrated parameters, propagation of surrogate error into the inverse solution, and posterior
predictive checks all remain open methodological questions. Finally, field validation in
unconventional reservoirs remains scarce despite the fact that these systems exhibit exactly the
kind of strong multiphysics coupling, limited observability, and parameter nonuniqueness for which
surrogate-assisted inverse adaptation would be most valuable.

Overall, the literature supports three main conclusions. First, field-validated mechanistic modeling
of fracture flowback already exists and has been demonstrated on real wells, but repeated use of
such models for calibration or optimization remains computationally prohibitive. Second, petroleum
surrogate modeling is well developed as a forward-emulation strategy, yet existing studies
overwhelmingly treat the proxy as a standalone accelerator rather than as the engine of inverse
calibration against field data. Third, the combination of a field-validated mechanistic basis,
dimensionless surrogate reduction, interpretable sensitivity structure, and decomposed inverse
adaptation has not been established as an integrated workflow for fracture flowback. The proposed
approach is therefore motivated not by the novelty of surrogate models or history matching
individually, but by their sequential coupling into a coherent pipeline that retains physical
grounding while making field-oriented calibration computationally feasible.

---

## New References Added (not in Galliamov et al. 2026)

**Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019).** Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial differential
equations. *Journal of Computational Physics*, 378, 686–707.
→ Used as contrast: PINNs embed PDEs in the loss (physics-constrained), our approach emulates a
validated simulator output (physics-emulating). The distinction justifies our choice.

**Bhosekar, A., Ierapetritou, M. (2018).** Advances in surrogate based modeling, feasibility
analysis, and optimization: A review. *Computers & Chemical Engineering*, 108, 250–267.
→ Broadens surrogate context beyond petroleum; supports the claim that dimensionless input
reduction is beneficial for surrogate robustness and inverse tractability.

**Razavi, S., Tolson, B.A., Burn, D.H. (2012).** Review of surrogate modeling in water resources.
*Water Resources Research*, 48, W07401.
→ Foundational cross-domain surrogate review. Shows that surrogate methodology is not
petroleum-specific — supports the universality argument in Section 2.3.

**Gholami, R., Shahraki, A.R., Jamali Paghaleh, M. (2019).** (Smart proxy for CO₂-EOR)
→ Already present in Gleb's version; kept. Not cited in original Galliamov et al. 2026 paper.

**Lam, R. et al. (2023).** Learning skillful medium-range global weather forecasting. *Science*
382, 1416–1421. → Already in Gleb's version; cross-domain analogy. Not in original paper.

**Bi, K. et al. (2023).** Accurate medium-range global weather forecasting with 3D neural networks.
*Nature* 619, 533–538. → Already in Gleb's version; cross-domain analogy. Not in original paper.

---

## References Retained from Galliamov et al. 2026 (overlap — necessary, these are the must-cites)

These overlap with the original paper but MUST remain in the joint paper because they establish
the physical basis — removing them would be incorrect attribution, not a solution to self-plagiarism.
The self-plagiarism risk is in TEXT, not in citations.

- Osiptsov et al. 2020 — mechanistic model base
- Boronin et al. 2023 — integrated modeling, field validation
- Vainshtein et al. 2020, 2021 — field testing
- Kalyuzhnyuk et al. 2019 — nearest prior art for HF neural surrogate
- Moradi et al. 2022 — completion design proxy
- Morozov et al. 2020 — data-driven contrast case
- Wang et al. 2021 — production forecasting surrogate
- Merzoug & Rasouli 2023 — nearest to combined approach

**NOTE:** Galliamov et al. (2026) must be added to the reference list. It does not yet appear in
Gleb's docx. This is the most important fix — the joint paper must cite Artem's original paper
as the source of the surrogate model and its validation results.
