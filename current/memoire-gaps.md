# Memoire Gap Analysis: Stage-1 vs Full-Mechanism Scope Mismatch

**Top-Priority Cross-Cutting Issue:**  
The Introduction and Method drafts narrate the *full Stage-5 dopamine handoff* (W_eff = f(DA)·W, the DA-request neuron, hypotheses H2–H5 with pass/fail thresholds) as the study's payload. [[PROJECT_STATUS_AND_PLAN]] commits only to *Stage 1* (single 256-unit CTRNN, working-memory attractor, pipeline validation), with the DA handoff explicitly "planned/future work." 

**Mandatory structural fix:** The entire mémoire must consistently separate *proposed mechanism* (the full dopamine hypothesis) from *demonstrated result* (Stage 1: working-memory recurrence + APE baseline). The Method contradicts itself (§2.7 claims "H1, H2, and H7" are the minimum sufficient result, but H2 requires the Stage-5 mechanism that is future work).

---

## Introduction Gaps

### Missing Core Arguments

1. **Outline 1.6 (local vs not-global prediction errors) absent as a distinct argument.**  
   - The project's own self-description flags local/not-global prediction errors as the MOST ORIGINAL thesis.  
   - Currently APE appears only as the habit-learning rule; the adaptive prediction element (how the network learns error structure per context) lacks dedicated prose.  
   - **Action:** Strengthen 1.6 as a two-sentence claim with [[citekey]] support.

2. **Outline 1.7 (supervisor's low-rank/attractor fork) entirely absent.**  
   - This is a stated prerequisite to finalising framing.  
   - **Action:** Either incorporate the fork as a "alternative mechanism" subsection or formally document why it was rejected.

### Citation Gaps & Flags

3. **Devaluation foundations uncited.**  
   - Jury expects [[adams1981]], [[dickinson1985]], [[yin2006]] to ground why reward devaluation is a meaningful proxy for dopamine function.  
   - **Action:** Add one paragraph (1.3 or 1.4) citing these.

4. **OpAL ([[frank2005]]) and Yaghoubi/Echeveste ([[yaghoubi2026]]) flagged "to acquire."**  
   - Both are load-bearing references for actor-critic architectures and online plasticity.  
   - **Action:** Confirm in Zotero; if present, cite and quote. If absent, prioritize acquisition.

5. **DA-wave citations ([[hamid2021a]], [[engel2024]]) flagged for removal.**  
   - Rationale: This model uses a single scalar dopamine signal, not spatial waves.  
   - **Action:** Remove or reframe to "empirical DA dynamics not captured in this simplified model."

6. **Prose-named citations lack [[citekey]] anchors.**  
   - Example: "Frank et al. showed..." should be "[[frank2005]] showed..." for wikilink consistency.  
   - **Action:** Audit all author-name cites in Introduction; convert to wikilinks.

---

## Method Gaps

### Coherence Risk: DA-Request Training (§2.3.1)

7. **Unresolved [OPEN] decision on DA-request supervision.**  
   - The outline specifies two contradictory paths: (a) reward-supervised learning, or (b) local prediction-error supervision.  
   - *Reward-supervised* is circular (reward is what we measure, not what trains the DA signal).  
   - *Local-PE-supervised* requires specifying what error the DA neuron learns to predict (error *in what space?*).  
   - **Action:** Resolve this in §2.3.1 with explicit design choice + justification. This is not a TODO; it's a methodological linchpin.

### Under-Described Architecture

8. **Complexity ladder (outline 2.3) under-described.**  
   - Outline sketches Models A, B, C, D with increasing sophistication.  
   - Method section offers only the Stage-1 baseline (single CTRNN).  
   - **Action:** Add a subsection (§2.1 or early §2.3) listing the progression: why each complexity step was chosen, what each tests, and which hypotheses require which model.

9. **Batched and configurable-observation environment (outline 2.2) under-described.**  
   - How are observations batched? What configurations are tested?  
   - **Action:** Expand environment design with concrete numbers (batch size, observation dimensionality, context-switch frequency).

### Relegation of Stage-1 Headline Analysis

10. **Fixed-point finder and participation ratio in Annex only.**  
    - These *are* the Stage-1 headline analysis per [[PROJECT_STATUS_AND_PLAN]].  
    - Currently they live in Annex; they should be in §2.6 or §2.7 (main Method).  
    - **Action:** Promote fixed-point and participation-ratio descriptions to the Method body; justify why each is essential for H1/H6/H7 validation.

### Internal Inconsistency (§2.7)

11. **§2.7 claims "H1, H2, and H7" are the minimum sufficient result.**  
    - But [[PROJECT_STATUS_AND_PLAN]] places H2 (the DA handoff) in Stage 5, not Stage 1.  
    - Stage 1 tests only H1, H6, H7 (working memory, APE baseline, fixed-point stability).  
    - **Action:** Revise §2.7 to state: "H1, H6, and H7 are the Stage-1 minimum result. H2–H5 require Stage 5 (future work)."

---

## Sections Not Yet Started

### Results

12. **Results section must state Stage-1 scope explicitly.**  
    - Organize subsections by Stage-1 hypotheses: H1 (working-memory attractor), H6 (APE rule learned), H7 (fixed-point stability).  
    - **Action:** For each hypothesis, present (a) supporting [[citekey]], (b) observed network behavior, (c) quantitative metrics (e.g., participation ratio, recurrent coupling strength), (d) comparison to baseline/control.  
    - Do NOT present H2–H5 results (the DA handoff is future work).

### Discussion

13. **Discussion must separate Stage-1 findings from future Stage-5 hypothesis.**  
    - Subsections: (a) Stage-1 validates working-memory + APE learning; (b) Stage-5 plan (DA handoff) requires additional mechanisms not yet tested.  
    - **Action:** For each H1/H6/H7 result, relate to literature ([[frank2005]], [[yin2006]], [[suri2002]]) and explain why the result constrains the full dopamine story.  
    - Flag explicitly: "Testing H2 requires the DA-request neuron (§2.3.1) and reward-devaluation paradigm (Stage 5)."

### Conclusion

14. **Conclusion must restate the scope and next steps.**  
    - Claim: "We demonstrate working-memory recurrence + APE learning in a Stage-1 CTRNN. This validates two of seven hypotheses (H1, H6, H7) and is necessary (but not sufficient) for the full dopamine handoff (H2–H5, Stage 5)."  
    - **Action:** One paragraph on why these Stage-1 results are valuable; one paragraph on the Stage-5 roadmap.

### Abstract

15. **Abstract must flag the scope.**  
    - Lead with: "We propose a seven-hypothesis dopamine model for working-memory credit assignment."  
    - Then: "Here we focus on Stage 1: validating working-memory recurrence and adaptive prediction-error learning in a 256-unit recurrent neural network."  
    - Close with: "Stage 5 tests the full dopamine handoff via reward devaluation (future work)."  
    - **Action:** Draft a 150-word abstract with Stage-1/Stage-5 labeling throughout.

---

## Cross-Cutting Issues

### 1. Scope Mismatch (CRITICAL)

- **Problem:** Introduction narrates full H2–H5; Method/Plan restrict to Stage 1.  
- **Fix:** Every section must label which hypotheses are tested now (H1, H6, H7) and which are future (H2–H5). Update Introduction §1.5 to foreshadow this structure.

### 2. §2.7 Internal Inconsistency

- **Problem:** Claims "H1, H2, H7" sufficient; [[PROJECT_STATUS_AND_PLAN]] says H2 is Stage 5.  
- **Fix:** Revise to "H1, H6, H7 are the Stage-1 minimum."

### 3. Supervisor Fork Not Resolved

- **Problem:** Outline 1.7 (low-rank/attractor mechanism) is a stated prerequisite to finalising framing; it is absent from drafts.  
- **Impact:** Framing cannot be signed off until this fork is either incorporated or rejected.  
- **Action:** Schedule explicit discussion with supervisor to resolve; document decision in Method/Introduction.

### 4. Dopamine Hypothesis Not Grounded in Result

- **Problem:** The full dopamine story (H2–H5) is proposed but not tested; H1, H6, H7 do not require dopamine.  
- **Impact:** Jury may ask "Why dopamine? Why not any other neuromodulator?"  
- **Fix:** In Introduction, cite literature anchoring dopamine specifically to working-memory credit assignment ([[frank2005]], [[suri2002]], [[yin2006]]) *and* flag that the dopamine test (devaluation, Stage 5) is future work.

### 5. Citation Hygiene & Library Gaps

- **Missing [[citekey]] anchors:** All prose-named citations ("Frank et al. showed...") must become wikilinks ([[frank2005]]).  
- **Flagged for acquisition:** [[frank2005]], [[yaghoubi2026]] — confirm in Zotero; if absent, prioritize.  
- **Flagged for removal:** [[hamid2021a]], [[engel2024]] — single scalar DA, not waves; reframe or delete.  
- **Action:** Audit Zotero; add missing keys; run citation consistency check (prose name → [[citekey]]).

### 6. Terminology Drift

- **Terms used inconsistently:** "DA-request" vs "DA-recruitment" vs "dopamine demand."  
- **Action:** Method §2.3.1 must define all three terms and use consistently. Recommend: "DA-request neuron" for the unit in H2; "dopamine recruitment" for the mechanism; "dopamine demand" for the signal value.

### 7. Criterion Clarification

- **Outline mentions "80%-not-90% criterion"** but scope and threshold are unclear.  
- **Action:** Clarify in Results: Is this a fixed-point participation ratio threshold? A learning-curve plateau? Add to §2.6 (quantitative metrics).

---

## Summary Checklist

- [ ] **Scope:** Revise Introduction §1.5, all of Results, Discussion, and Conclusion to explicitly separate Stage-1 (H1, H6, H7) from Stage-5 (H2–H5).  
- [ ] **Internal consistency:** Fix §2.7 claim to "H1, H6, H7 minimum."  
- [ ] **Missing arguments:** Add Outline 1.6 as a dedicated subsection; resolve Outline 1.7 fork.  
- [ ] **DA-request design:** Resolve the reward-supervised vs local-PE-supervised decision in §2.3.1.  
- [ ] **Architecture description:** Expand complexity ladder and environment design.  
- [ ] **Promote Stage-1 analysis:** Move fixed-point and participation-ratio to Method body.  
- [ ] **Citations:** Audit all prose names, add [[citekey]] anchors, confirm Zotero library, remove wave-based DA refs.  
- [ ] **Devaluation foundations:** Add [[adams1981]], [[dickinson1985]], [[yin2006]] to Introduction 1.3 or 1.4.  
- [ ] **Supervisor fork:** Schedule resolution discussion; document in Introduction.  
- [ ] **Terminology:** Standardize DA-request/recruitment/demand in §2.3.1.  
- [ ] **Results/Discussion/Conclusion:** Draft with Stage-1 scope and forward reference to Stage-5 roadmap.  
- [ ] **Abstract:** 150-word draft with Stage-1/Stage-5 labeling.  
- [ ] **80%-not-90% criterion:** Clarify and place in Results metrics.
