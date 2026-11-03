import collections
import os
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate
from torch import nn
import torch.nn.functional as F
import wandb

from .base import BasePipeline
from utils.misc.misc import (
    get_mask,
    get_remain_time,
    load_file,
)


class CRAPipeline(BasePipeline):
    """
    CRA R2-C pipeline with V3.7 masked language pooling and Event-Relative
    Grounding. The V3.5 Q1/Q2 objective, QA/support gradient separation,
    fixed support graph, and validation-only checkpoint selection are retained.

    Q-Bound R2-A/B retain three optimization routes; R2-C has two:

        A/B: L_CRA/main + lambda_G * L_ground + lambda_P * L_pred
        C:   L_CRA/main + lambda_G * L_ground

    * L_CRA/main consumes the QA feature pooled with an identical but detached
      Support Field.  It cannot train temporal relevance or extent topology.
    * L_ground aligns differentiable raw-CLIP evidence with a detached,
      question-only target.  The structural graph is detached, so this route
      trains semantic relevance only.
    * V3.5 optionally adds SAME-VIDEO Q1/Q2 evidence discrimination on top.
      It does not modify Q1 QA, R2-C L_ground, fixed propagation, or decoder.
    * A/B use L_pred to train an independent video-only predictor.
      C has no predictor and uses fixed adjacent edges directly.

    The Question creates a semantic seed; a detached video-only predictive
    graph propagates that seed into support_temporal_prob.  The same numerical
    support is used for QA, grounding, training, and inference.  Analytic
    boundary/posterior-barycenter spans are metadata and evaluation views only:
    there is no hard-MAP temporal decision or learned span scorer.
    """

    def __init__(self, cfgs) -> None:
        # Test metrics must be observed and logged to WandB EVERY epoch. Set
        # this before BasePipeline initialization because Base also reads it.
        # Only validation metrics are allowed to select a checkpoint.
        stat_cfg = cfgs.get("stat")
        if not isinstance(stat_cfg, dict):
            raise TypeError("stat must be a mapping/dict.")
        if stat_cfg.get("eval_test_each_epoch", True) is not True:
            raise ValueError(
                "V3.7 requires stat.eval_test_each_epoch=true; "
                "do not disable per-epoch test logging."
            )
        stat_cfg.setdefault("eval_test_each_epoch", True)
        super().__init__(cfgs)

        self.qbound_version = "2.8.0"
        self.r2_graph_mode = (
            self._qbound_config().get("support_graph_mode", "learned")
            if self._qbound_enabled() else None
        )
        self.r2_predictor_enabled = self._qbound_config().get("support_predictor_enabled", True)
        if not isinstance(self.r2_predictor_enabled, bool):
            raise TypeError("support_predictor_enabled must be boolean.")
        if self._qbound_enabled() and not self.r2_predictor_enabled and self.r2_graph_mode != "fixed":
            raise ValueError("R2-C requires fixed graph.")
        self.eval_test_each_epoch = self.cfgs["stat"].get(
            "eval_test_each_epoch", True
        )
        self.save_val_predictions = self.cfgs["stat"].get(
            "save_val_predictions", True
        )
        for name, value in (
            ("stat.eval_test_each_epoch", self.eval_test_each_epoch),
            ("stat.save_val_predictions", self.save_val_predictions),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        # Q-Bound core/decoder still advertise 2.8.0 for checkpoint compatibility.
        # V3.5 is a training experiment identifier, NOT a changed core version.
        self.v35_pair_settings = self._v35_pair_loss_settings()
        self._v35_loss_step = 0
        # H1 representation intervention. Keep losses/sampler/decoder unchanged.
        q_cfg = self._qbound_config() if self._qbound_enabled() else {}
        self.h1_temporal_mode = (
            q_cfg.get("temporal_context_mode", "eq_ptc")
            if self._qbound_enabled() else None
        )
        if self._qbound_enabled() and self.h1_temporal_mode not in ("legacy", "eq_ptc"):
            raise ValueError("H1 temporal_context_mode must be 'legacy' or 'eq_ptc'.")
        # V3.7 mode is checked against all three previously patched modules.
        # Default remains backward-compatible for legacy/EQ-PTC controls.
        self.v37_event_mode = (
            q_cfg.get("event_grounding_mode", "legacy")
            if self._qbound_enabled() else "legacy"
        )
        if self.v37_event_mode not in ("legacy", "event_relative"):
            raise ValueError("V3.7 event_grounding_mode must be 'legacy' or 'event_relative'.")
        pooling_default = "masked" if self.v37_event_mode == "event_relative" else "legacy"
        self.v37_pooling_mode = (
            q_cfg.get("language_pooling_mode", pooling_default)
            if self._qbound_enabled() else "legacy"
        )
        if self.v37_pooling_mode not in ("legacy", "masked"):
            raise ValueError("V3.7 language_pooling_mode must be 'legacy' or 'masked'.")
        self.v37_event_max_length = q_cfg.get("event_candidate_max_length", None)
        if self.v37_event_max_length is not None and (
            isinstance(self.v37_event_max_length, bool)
            or not isinstance(self.v37_event_max_length, int)
            or self.v37_event_max_length < 1
        ):
            raise ValueError("V3.7 event_candidate_max_length must be null or int >= 1.")
        self._h1_check_optimizer_contract()
        self._v37_check_optimizer_contract()
        if self.v35_pair_settings["enabled"]:
            if not self._qbound_enabled() or self.r2_graph_mode != "fixed" or self.r2_predictor_enabled:
                raise ValueError("QDTG V3.5 requires enabled R2-C with fixed graph and no predictor.")

        if self._qbound_enabled():
            model_cfg = self.cfgs.get("model", {})
            qbound_cfg = model_cfg.get("qbound", {})
            if not isinstance(qbound_cfg, dict):
                raise TypeError("model.qbound must be a mapping/dict.")

            configured_version = str(
                qbound_cfg.get(
                    "version",
                    model_cfg.get("qbound_version", ""),
                )
            )
            if configured_version != self.qbound_version:
                raise ValueError(
                    "CRAPipeline/Q-Bound version mismatch: expected "
                    f"model.qbound.version={self.qbound_version!r}, got "
                    f"{configured_version!r}."
                )

            stale_keys = (
                "functional_critic",
                "boundary_refiner",
                "relation_weight",
                "boundary_weight",
                "adaptive_radius_ratio",
                "min_radius",
                "functional_composition",
                "boundary_equilibrium",
                "answer_only_supervision",
                "visual_continuity_weight",
                "question_continuity_weight",
                "continuity_mix",
                "continuity_gate",
                "closure_temperature",
                "closure_weight",
                "core_weight",
                "exact_bottleneck_closure",
                "visual_distinctiveness_weight",
                "visual_transition_weight",
                "query_routing_weight",
                "query_routing_temperature",
                "hard_map_temperature",
                "span_scorer",
                "event_candidate_selector",
            )
            stale = sorted(key for key in stale_keys if key in qbound_cfg)
            if stale:
                raise ValueError(
                    "Q-Bound V2.8 config contains removed legacy keys: "
                    + ", ".join(stale)
                )

    def _h1_core(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(model, "decoder", None)
        return getattr(decoder, "qbound", None)

    def _h1_check_optimizer_contract(self):
        """Fail closed on a silent EQ-PTC -> legacy integration or optimizer bypass.

        Called after BasePipeline construction. This is *not* a parameter-freezing
        policy: every active EQ-PTC parameter must be included in the optimizer.
        """
        if not self._qbound_enabled():
            return
        core = self._h1_core()
        if core is None:
            raise RuntimeError("H1: CRA/RefineDecoder did not instantiate QBoundCore.")
        actual = getattr(core, "temporal_context_mode", None)
        if actual != self.h1_temporal_mode:
            raise RuntimeError(
                f"H1 architecture mismatch: configured={self.h1_temporal_mode!r}, "
                f"core={actual!r}. Update core.py/refine.py/cra.py together."
            )
        model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(model, "decoder", None)
        if getattr(decoder, "qbound_temporal_context_mode", None) != self.h1_temporal_mode:
            raise RuntimeError("H1: RefineDecoder is not using the configured temporal mode.")
        if self.h1_temporal_mode == "eq_ptc":
            if self.r2_graph_mode != "fixed" or self.r2_predictor_enabled:
                raise RuntimeError("H1 EQ-PTC must retain R2-C fixed, predictor-free graph.")
            group_ids = {
                id(p) for group in self.optimizer.param_groups
                for p in group["params"]
            }
            named = {
                n: p for n,p in core.named_parameters()
                if n.startswith("eqptc_")
            }
            if not named:
                raise RuntimeError("H1: zero EQ-PTC parameters; wrong core.py?")
            missing = [n for n,p in named.items() if not p.requires_grad or id(p) not in group_ids]
            if missing:
                raise RuntimeError(
                    "H1: new EQ-PTC parameters are frozen or absent from optimizer: "
                    + ", ".join(missing[:12])
                )
            # Keep optimizer's existing Q-Bound learning-rate policy. A separate
            # LR or loss reweighting would introduce a second experimental change.

    def _v37_check_optimizer_contract(self):
        """Verify V3.7 is really active and every comparator parameter trains.

        This is a fail-fast guard, NOT a change to optimizer groups/LR policy.
        BasePipeline remains responsible for constructing the optimizer.
        """
        if not self._qbound_enabled():
            return
        core = self._h1_core()
        model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(model, "decoder", None)
        if core is None or decoder is None:
            raise RuntimeError("V3.7: missing QBoundCore/RefineDecoder.")
        for label, actual in (
            ("CRA", getattr(model, "qbound_event_grounding_mode", None)),
            ("RefineDecoder", getattr(decoder, "qbound_event_grounding_mode", None)),
            ("QBoundCore", getattr(core, "event_grounding_mode", None)),
        ):
            if actual != self.v37_event_mode:
                raise RuntimeError(
                    f"V3.7 event mode mismatch: {label}={actual!r}, "
                    f"YAML={self.v37_event_mode!r}."
                )
        for label, actual in (
            ("CRA", getattr(model, "qbound_language_pooling_mode", None)),
            ("LanModel", getattr(getattr(model, "lang_model", None), "pooling_mode", None)),
        ):
            if actual != self.v37_pooling_mode:
                raise RuntimeError(
                    f"V3.7 pooling mode mismatch: {label}={actual!r}, "
                    f"YAML={self.v37_pooling_mode!r}."
                )
        if getattr(core, "event_candidate_max_length", "missing") != self.v37_event_max_length:
            raise RuntimeError("V3.7 event_candidate_max_length mismatch.")
        named = {name: p for name, p in core.named_parameters()
                 if name.startswith("event_comparator.")}
        if self.v37_event_mode == "legacy":
            if named:
                raise RuntimeError("V3.7 legacy mode unexpectedly instantiated Event Comparator.")
            return
        if self.h1_temporal_mode != "eq_ptc" or self.r2_graph_mode != "fixed" or self.r2_predictor_enabled:
            raise RuntimeError("V3.7 event_relative requires EQ-PTC + predictor-free R2-C.")
        if not named:
            raise RuntimeError("V3.7: Event Comparator parameters are absent.")
        group_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        missing = [name for name, p in named.items()
                   if not p.requires_grad or id(p) not in group_ids]
        if missing:
            raise RuntimeError(
                "V3.7 Event Comparator is frozen or absent from optimizer: "
                + ", ".join(missing[:12])
            )

    def _v37_check_runtime(self, gqa_output):
        """Check candidate scores, residual Seed path and Q2 independently.

        This is an inference/training invariant check; it never reads GT.
        """
        if not self._qbound_enabled():
            return
        aux = self._get_qbound_aux(gqa_output)
        if not isinstance(aux, dict):
            raise RuntimeError("V3.7: Q-Bound auxiliary outputs are missing.")
        if aux.get("event_grounding_mode") != self.v37_event_mode:
            raise RuntimeError("V3.7: forward did not execute configured event mode.")
        flag = aux.get("event_relative_enabled_flag")
        if not torch.is_tensor(flag) or flag.numel() != 1 or (
            float(flag.detach()) != float(self.v37_event_mode == "event_relative")
        ):
            raise RuntimeError("V3.7: Event Comparator activation flag is incorrect.")
        if self.v37_event_mode != "event_relative":
            if "event_candidate_prob" in aux:
                raise RuntimeError("V3.7 legacy mode should not score event candidates.")
            return

        video_mask = aux["video_mask"].bool()
        B, T = video_mask.shape
        starts = aux["event_candidate_starts"]
        ends = aux["event_candidate_ends"]
        lengths = aux["event_candidate_lengths"]
        candidate_mask = aux["event_candidate_valid"]
        K = starts.numel()
        max_length = self.v37_event_max_length or T
        expected_K = sum(T - length + 1 for length in range(1, min(max_length, T) + 1))
        if K != expected_K or candidate_mask.shape != (B, K):
            raise RuntimeError(
                f"V3.7 candidate bank shape mismatch: K={K}, expected {expected_K}."
            )
        if candidate_mask.dtype != torch.bool:
            raise TypeError("V3.7 candidate_valid must be bool.")
        if not torch.equal(lengths, ends - starts + 1):
            raise RuntimeError("V3.7 candidate boundary/length mismatch.")
        for name, expected_shape in (
            ("event_candidate_scores", (B, K)),
            ("event_candidate_prob", (B, K)),
            ("event_frame_prob", (B, T)),
            ("event_geometry_frame_prior", (B, T)),
            ("event_seed_residual_logits", (B, T)),
            ("legacy_semantic_seed_prob", (B, T)),
            ("legacy_semantic_seed_logits", (B, T)),
            ("event_candidate_top_index", (B,)),
            ("event_candidate_top_start_frame", (B,)),
            ("event_candidate_top_end_frame", (B,)),
            ("event_candidate_count", (B,)),
            ("event_residual_rms", (B,)),
            ("event_residual_to_legacy_std_ratio", (B,)),
        ):
            value = aux.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                raise RuntimeError(f"V3.7: missing/invalid {name} shape {expected_shape}.")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"V3.7: {name} has NaN/Inf.")
        prob = aux["event_candidate_prob"]
        if (prob.detach() < -1e-6).any() or (
            (prob.detach() * ~candidate_mask).abs().amax() > 1e-5
        ) or not torch.allclose(prob.detach().sum(-1), torch.ones(B, device=prob.device,
                                dtype=prob.dtype), atol=1e-4, rtol=1e-4):
            raise RuntimeError("V3.7: event candidate distribution is invalid.")
        if not torch.equal(aux["event_candidate_count"], candidate_mask.sum(-1)):
            raise RuntimeError("V3.7: event candidate count mismatch.")
        if not torch.equal(aux["event_candidate_top_index"], prob.detach().argmax(-1)):
            raise RuntimeError("V3.7: event top candidate mismatch.")
        for name in ("event_frame_prob", "event_geometry_frame_prior",
                     "legacy_semantic_seed_prob"):
            values = aux[name].detach()
            if ((values * ~video_mask).abs().amax() > 1e-5
                or not torch.allclose((values * video_mask).sum(-1),
                                      torch.ones(B, dtype=values.dtype, device=values.device),
                                      atol=1e-4, rtol=1e-4)):
                raise RuntimeError(f"V3.7: {name} invalid on video padding.")
        residual = aux["event_seed_residual_logits"]
        if (residual.detach() * ~video_mask).abs().amax() > 1e-5:
            raise RuntimeError("V3.7: residual assigns signal to invalid frames.")
        if self.model.training and torch.is_grad_enabled():
            if not prob.requires_grad or not aux["support_temporal_prob"].requires_grad:
                raise RuntimeError("V3.7 Event Comparator is detached from Support.")

        pair = aux.get("qbound_pair_aux")
        if pair is not None:
            if not isinstance(pair, dict):
                raise TypeError("V3.7 qbound_pair_aux must be a dictionary.")
            # Core.forward_paired() deliberately shares Q1 interval geometry,
            # but DOES NOT duplicate it into Q2's output dictionary.
            # Q2 reports event_candidate_count and event_candidate_prob instead.
            count2 = pair.get("event_candidate_count")
            if not torch.is_tensor(count2) or not torch.equal(
                count2, aux["event_candidate_count"]
            ):
                raise RuntimeError("V3.7 Q1/Q2 candidate counts differ.")
            prob2 = pair.get("event_candidate_prob")
            if not torch.is_tensor(prob2) or (
                (prob2.detach() * ~candidate_mask).abs().amax() > 1e-5
            ):
                raise RuntimeError("V3.7 Q2 assigns mass to invalid shared candidates.")
            for name, shape in (
                ("event_candidate_scores", (B, K)),
                ("event_candidate_prob", (B, K)),
                ("event_frame_prob", (B, T)),
                ("event_seed_residual_logits", (B, T)),
            ):
                value = pair.get(name)
                if not torch.is_tensor(value) or value.shape != shape or not torch.isfinite(value).all():
                    raise RuntimeError(f"V3.7 Q2 missing/invalid {name}.")
            if self.model.training and torch.is_grad_enabled() and not pair["event_candidate_prob"].requires_grad:
                raise RuntimeError("V3.7 Q2 comparator has no gradient path.")

    def _h1_check_runtime(self, gqa_output):
        """Check EQ-PTC activation and evidence shape at training AND validation."""
        if not self._qbound_enabled():
            return
        aux = self._get_qbound_aux(gqa_output)
        if not isinstance(aux, dict):
            raise RuntimeError("H1: missing qbound_aux in VQA outputs.")
        actual_mode = aux.get("temporal_context_mode")
        if actual_mode != self.h1_temporal_mode:
            raise RuntimeError(
                f"H1 run mode mismatch: config={self.h1_temporal_mode!r}, "
                f"VQA output={actual_mode!r}."
            )
        flag = aux.get("eqptc_enabled_flag")
        if not torch.is_tensor(flag) or flag.numel()!=1 or float(flag.detach()) != float(self.h1_temporal_mode=="eq_ptc"):
            raise RuntimeError("H1 EQ-PTC enabled flag does not match config.")
        if self.h1_temporal_mode == "eq_ptc":
            mask = aux["video_mask"]
            qmask = aux["question_mask"]
            core = self._h1_core()
            b,t = mask.shape
            h = core.num_question_heads
            d = core.hidden_dim
            required = {
                "eqptc_global_context_states": (b,t,d),
                "eqptc_conditioned_visual_states": (b,t,d),
                "eqptc_early_question_attention": (b,h,t,qmask.size(1)),
                "eqptc_temporal_attention": (b,h,t,t),
                "eqptc_early_delta_ratio": (b,),
                "eqptc_temporal_delta_ratio": (b,),
            }
            for key,shape in required.items():
                value = aux.get(key)
                if not torch.is_tensor(value) or tuple(value.shape)!=shape:
                    raise RuntimeError(f"H1 output {key} has wrong/missing shape {shape}.")
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"H1 output {key} contains NaN/Inf.")

    @staticmethod
    def _h1_gt_evidence_metrics(seed_by_id, support_by_id, gt, timestamps,
                                question_csv_path=None):
        """Validation-only H1 diagnostics; never enter get_loss/training.

        GT semantics match H1-01: a sampled timestamp is inside ANY annotated
        [start,end] interval, including endpoints. Paired GT-swap is computed
        only for same-video pairs whose sampled GT masks are disjoint.
        """
        if not seed_by_id or not support_by_id:
            return {}
        import itertools
        # EXACT exploratory H1-02 lexical cohort, NOT substring matching:
        # sklearn English stopword removal + word (1,2)-gram TF-IDF fitted on
        # the complete question CSV of the current evaluation split.
        # The same cohort is used for legacy and EQ-PTC and has NO loss path.
        tfidf_matrix = None
        tfidf_index = {}
        if question_csv_path is not None:
            if not os.path.isfile(question_csv_path):
                raise FileNotFoundError(f"H1 question CSV is missing: {question_csv_path}")
            import csv
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
            except ImportError as exc:
                raise ImportError(
                    "H1 TF-IDF cohort requires scikit-learn for reproducibility."
                ) from exc
            question_corpus = []
            with open(question_csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("video_id") and row.get("qid") and row.get("question"):
                        qid = f"{row['video_id']}_{row['qid']}"
                        if qid in tfidf_index:
                            # Duplicate identifiers cannot unambiguously map to
                            # a particular validation question: fail closed.
                            raise ValueError(f"H1 duplicate evaluation question id: {qid}")
                        tfidf_index[qid] = len(question_corpus)
                        question_corpus.append(str(row["question"]))
            if question_corpus:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(question_corpus)
        grouped = collections.defaultdict(list)
        coverage = []
        seeded = []
        skipped = 0
        no_sample_gt = 0
        for qid, prob_list in seed_by_id.items():
            text_id = str(qid)
            if "_" not in text_id:
                skipped += 1
                continue
            vid, local_id = text_id.split("_",1)
            video_meta = gt.get(vid)
            frame_times = timestamps.get(vid)
            if not isinstance(video_meta,dict) or frame_times is None:
                skipped += 1
                continue
            intervals = video_meta.get("location",{}).get(local_id)
            if not intervals or qid not in support_by_id:
                skipped += 1
                continue
            ts = np.asarray(frame_times,dtype=np.float64).reshape(-1)
            seed = np.asarray(prob_list,dtype=np.float64).reshape(-1)
            support = np.asarray(support_by_id[qid],dtype=np.float64).reshape(-1)
            if len(ts)!=len(seed) or len(seed)!=len(support):
                raise ValueError(f"H1 timestamps/Seed/Support mismatch for {qid!r}.")
            if not np.isfinite(seed).all() or not np.isfinite(support).all():
                raise RuntimeError(f"H1 non-finite distributions for {qid!r}.")
            mask = np.zeros(len(ts),dtype=bool)
            for start,end in intervals:
                mask |= (ts>=float(start)) & (ts<=float(end))
            if not np.any(mask):
                # No sampled GT frame. Include zero for FULL-set performance,
                # but exclude from paired GT density and identifiable subset.
                no_sample_gt += 1
                continue
            coverage.append(mask.mean())
            entry={"qid":text_id,"seed":seed,"support":support,"mask":mask,
                   "peak":int(seed.argmax())}
            seeded.append(entry)
            grouped[vid].append(entry)
        if not seeded:
            return {"H1EligibleQuestions":0.0,"H1MetadataSkipped":float(skipped)}
        eval_n = len(seeded) + no_sample_gt
        all_pairs=0
        correct=0
        equal_peak=0
        # Reproduce EXACT H1-02 TF-IDF high-similarity threshold (>=0.45).
        # Exploratory cohort only; never use it for checkpoint selection.
        high_n=high_correct=high_same_peak=0
        within_video_rates=[]
        for values in grouped.values():
            video_pos=0
            video_n=0
            for a,b in itertools.combinations(values,2):
                if np.any(a["mask"] & b["mask"]):
                    continue
                # Normalized by GT sampled-frame count; not duration-biased.
                matched = (a["seed"][a["mask"]].mean()
                           +b["seed"][b["mask"]].mean()
                           -a["seed"][b["mask"]].mean()
                           -b["seed"][a["mask"]].mean())
                video_n+=1
                all_pairs+=1
                positive=int(matched>0)
                correct+=positive
                video_pos+=positive
                same_peak=int(a["peak"]==b["peak"])
                equal_peak+=same_peak
                if tfidf_matrix is not None:
                    ia = tfidf_index.get(a["qid"])
                    ib = tfidf_index.get(b["qid"])
                    if ia is None or ib is None:
                        raise KeyError("H1 prediction is missing question text in evaluation CSV.")
                    similarity = float(tfidf_matrix[ia].multiply(tfidf_matrix[ib]).sum())
                    if similarity >= 0.45:
                        high_n += 1
                        high_correct += positive
                        high_same_peak += same_peak
            if video_n:
                within_video_rates.append(video_pos/video_n)
        return {
            "H1QuestionCount":float(eval_n),
            "H1EligibleQuestions":float(len(seeded)),
            "H1ZeroSampleGTCount":float(no_sample_gt),
            "H1MetadataSkipped":float(skipped),
            "H1GTFrameCoverage":float(sum(coverage)/eval_n),
            "H1SeedPeakInGT":float(sum(x["mask"][x["peak"]] for x in seeded)/eval_n),
            "H1SupportPeakInGT":float(sum(x["mask"][x["support"].argmax()] for x in seeded)/eval_n),
            "H1SeedGTMass":float(sum(x["seed"][x["mask"]].sum() for x in seeded)/eval_n),
            "H1SupportGTMass":float(sum(x["support"][x["mask"]].sum() for x in seeded)/eval_n),
            "H1HighConfidenceWrongPeak":float(sum(
                (not x["mask"][x["peak"]]) and x["seed"].max()>0.8 for x in seeded)/eval_n),
            "H1DisjointPairCount":float(all_pairs),
            "H1DisjointPairGTMatch":float(correct/all_pairs) if all_pairs else 0.,
            "H1DisjointPairPeakCoincidence":float(equal_peak/all_pairs) if all_pairs else 0.,
            "H1VideoBalancedGTMatch":float(np.mean(within_video_rates)) if within_video_rates else 0.,
            "H1HighSimTFIDF45PairCount":float(high_n),
            "H1HighSimTFIDF45GTMatch":float(high_correct/high_n) if high_n else 0.,
            "H1HighSimTFIDF45PeakCoincidence":float(high_same_peak/high_n) if high_n else 0.,
        }

    def _v37_candidate_gt_diagnostics(
        self, top_spans, valid_frame_masks, gt, timestamps,
    ):
        """Sampled-frame candidate ranking/oracle metrics (EVAL ONLY).

        A candidate is a contiguous [s,e] interval. Oracle over the same
        candidate bank is a diagnostic upper bound, not a trainable target.
        IoU here is on sampled-frame masks, NOT the official seconds-based IoU.
        """
        if not top_spans:
            return {}
        evaluated, top_ious, oracle_ious, top_hits, oracle_hits = 0, [], [], [], []
        absent_gt = 0
        metadata_skipped = 0
        for qid, span in top_spans.items():
            text_id = str(qid)
            if "_" not in text_id:
                raise ValueError(f"V3.7 candidate diagnostic has invalid question id {qid!r}.")
            vid, local_id = text_id.split("_", 1)
            meta = gt.get(vid)
            ts_value = timestamps.get(vid)
            if not isinstance(meta, dict) or ts_value is None:
                metadata_skipped += 1
                continue
            intervals = meta.get("location", {}).get(local_id)
            if not intervals:
                # Mirror existing H1 eval policy: questions without GT are
                # excluded from this diagnostic, not fabricated or fatal.
                metadata_skipped += 1
                continue
            ts = np.asarray(ts_value, dtype=np.float64).reshape(-1)
            valid = np.asarray(valid_frame_masks[qid], dtype=bool).reshape(-1)
            if ts.size != valid.size:
                raise ValueError(f"V3.7 timestamps/valid mask length mismatch at {qid!r}.")
            gt_mask = np.zeros(ts.size, dtype=bool)
            for start, end in intervals:
                gt_mask |= (ts >= float(start)) & (ts <= float(end))
            gt_mask &= valid
            if not gt_mask.any():
                absent_gt += 1
                continue
            t = valid.size
            starts, ends = np.triu_indices(t)
            if self.v37_event_max_length is not None:
                keep = (ends - starts + 1) <= self.v37_event_max_length
                starts, ends = starts[keep], ends[keep]
            valid_prefix = np.pad(valid.astype(np.int32).cumsum(), (1, 0))
            valid_c = (valid_prefix[ends + 1] - valid_prefix[starts]) == (ends - starts + 1)
            if not valid_c.any():
                raise RuntimeError("V3.7 candidate bank has no valid interval in evaluation.")
            gt_prefix = np.pad(gt_mask.astype(np.int32).cumsum(), (1, 0))
            intersect = gt_prefix[ends + 1] - gt_prefix[starts]
            union = (ends - starts + 1) + int(gt_mask.sum()) - intersect
            iou = np.divide(intersect, union, out=np.zeros_like(union, dtype=float),
                            where=union > 0)
            iou[~valid_c] = 0.0
            s_top, e_top = map(int, span)
            top_match = np.flatnonzero((starts == s_top) & (ends == e_top) & valid_c)
            if len(top_match) != 1:
                raise RuntimeError(f"V3.7 top candidate absent/invalid at {qid!r}.")
            evaluated += 1
            top_ious.append(float(iou[top_match[0]]))
            oracle_ious.append(float(iou.max()))
            top_hits.append(bool(intersect[top_match[0]] > 0))
            oracle_hits.append(bool((intersect[valid_c] > 0).any()))
        if evaluated == 0:
            return {"QBoundEventCandidateEligible": 0.0,
                    "QBoundEventCandidateZeroSampleGT": float(absent_gt),
                    "QBoundEventCandidateMetadataSkipped": float(metadata_skipped)}
        return {
            "QBoundEventCandidateEligible": float(evaluated),
            "QBoundEventCandidateZeroSampleGT": float(absent_gt),
            "QBoundEventCandidateMetadataSkipped": float(metadata_skipped),
            "QBoundEventCandidateTopSampledIoU": float(np.mean(top_ious)),
            "QBoundEventCandidateOracleSampledIoU": float(np.mean(oracle_ious)),
            "QBoundEventCandidateTopSampledHit": float(np.mean(top_hits)),
            "QBoundEventCandidateOracleSampledHit": float(np.mean(oracle_hits)),
            "QBoundEventCandidateRankingGap": float(np.mean(oracle_ious) - np.mean(top_ious)),
        }

    def _model_device(self):
        return next(self.model.parameters()).device

    def _qbound_enabled(self):
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})

        if isinstance(qbound_cfg, dict) and "enabled" in qbound_cfg:
            return bool(qbound_cfg["enabled"])

        return bool(
            model_cfg.get(
                "qbound_enabled",
                model_cfg.get("gpeb_enabled", False),
            )
        )

    def _qbound_handover_settings(self):
        """Compatibility helper: V2.8 has no CRA -> Q-Bound handover."""
        return 1.0, 0

    def _train_qbound_mix(
        self,
        epoch,
        batch_index,
        batches_per_epoch,
    ):
        del epoch, batch_index, batches_per_epoch
        return 1.0 if self._qbound_enabled() else 0.0

    def _get_qbound_aux(self, gqa_output):
        if not self._qbound_enabled():
            return None
        if gqa_output is None or len(gqa_output) < 3:
            return None

        model_outputs = gqa_output[2]
        if not isinstance(model_outputs, dict):
            return None

        time_param = model_outputs.get("time_param")
        if not isinstance(time_param, dict):
            return None

        return time_param.get(
            "qbound_aux",
            time_param.get("gpeb_aux"),
        )

    @staticmethod
    def _prepare_labels(answer_id, device):
        return answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)

    @staticmethod
    def _scalar_mean(values):
        if values is None:
            return None
        if not torch.is_tensor(values):
            return float(values)
        if values.numel() == 0:
            return 0.0
        return values.detach().float().mean().item()

    @staticmethod
    def _normalize_distribution(values, mask=None, eps=1e-8):
        values = values.clamp_min(0.0)
        if mask is not None:
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(1)
            values = values * mask.to(values.dtype)
        return values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)

    @classmethod
    def _js_divergence(cls, first, second, mask=None, eps=1e-8):
        first = cls._normalize_distribution(
            first, mask=mask, eps=eps
        )
        second = cls._normalize_distribution(
            second, mask=mask, eps=eps
        )
        middle = 0.5 * (first + second)

        first_kl = torch.sum(
            first
            * (
                torch.log(first.clamp_min(eps))
                - torch.log(middle.clamp_min(eps))
            ),
            dim=-1,
        )
        second_kl = torch.sum(
            second
            * (
                torch.log(second.clamp_min(eps))
                - torch.log(middle.clamp_min(eps))
            ),
            dim=-1,
        )
        return 0.5 * (first_kl + second_kl)

    @staticmethod
    def _answer_decision_margin(logits, labels):
        """
        z_y - logsumexp_{a != y}(z_a), for diagnostics only.
        """
        if logits.ndim != 2:
            raise ValueError(
                "Answer logits must be [B,C], got "
                f"{tuple(logits.shape)}."
            )
        if logits.size(-1) < 2:
            raise ValueError(
                "Answer discrimination requires >=2 candidates."
            )

        labels = labels.reshape(-1)
        if labels.numel() != logits.size(0):
            raise ValueError(
                "labels do not match answer-logit batch size."
            )

        gather_index = labels[:, None]
        correct = logits.gather(
            -1, gather_index
        ).squeeze(-1)

        wrong = logits.clone()
        wrong.scatter_(
            -1,
            gather_index,
            float("-inf"),
        )
        wrong_lse = torch.logsumexp(
            wrong,
            dim=-1,
        )
        return correct - wrong_lse

    def _qbound_v280_loss_settings(self):
        """Validate the two mandatory V2.8 auxiliary objectives."""
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        def _read_branch(name, default_temperature=None):
            branch = qbound_cfg.get(name, {})
            if branch is None:
                branch = {}
            if not isinstance(branch, dict):
                raise TypeError(
                    f"model.qbound.{name} must be a mapping/dict."
                )

            enabled = bool(branch.get("enabled", True))
            weight = float(branch.get("loss_weight", 1.0))
            if weight < 0.0:
                raise ValueError(
                    f"model.qbound.{name}.loss_weight must be >= 0."
                )
            if not enabled and weight != 0.0:
                raise ValueError(
                    f"model.qbound.{name}.enabled=false requires "
                    "loss_weight=0."
                )

            result = {"enabled": enabled, "weight": weight}
            if default_temperature is not None:
                temperature = float(
                    branch.get("temperature", default_temperature)
                )
                if temperature <= 0.0:
                    raise ValueError(
                        f"model.qbound.{name}.temperature must be > 0."
                    )
                result["temperature"] = temperature
            return result

        grounding = _read_branch("grounding_loss", default_temperature=0.07)
        predictive = _read_branch("predictive_loss")
        predictor_enabled = bool(getattr(self, "r2_predictor_enabled", True))

        if self._qbound_enabled():
            if not grounding["enabled"] or grounding["weight"] <= 0.0:
                raise ValueError(
                    "Canonical Q-Bound V2.8 requires a positive enabled "
                    "model.qbound.grounding_loss."
                )
            if predictor_enabled:
                if not predictive["enabled"] or predictive["weight"] <= 0.0:
                    raise ValueError("R2-A/B require positive L_pred.")
            elif predictive["enabled"] or predictive["weight"] != 0.0:
                raise ValueError("R2-C requires predictive_loss.enabled=false, loss_weight=0.")

        return {
            "grounding_enabled": grounding["enabled"],
            "grounding_weight": grounding["weight"],
            "grounding_temperature": grounding["temperature"],
            "predictive_enabled": predictive["enabled"],
            "predictive_weight": predictive["weight"],
        }

    def _score_answer_candidates(
        self,
        answer_proj,
        representation,
        detach_answer=True,
        normalize=False,
        logit_scale=1.0,
    ):
        if answer_proj.ndim != 3:
            raise ValueError(
                "answer_proj must be [B,C,D], got "
                f"{tuple(answer_proj.shape)}."
            )

        squeeze_dim = False
        if representation.ndim == 2:
            representation = representation.unsqueeze(1)
            squeeze_dim = True
        elif representation.ndim != 3:
            raise ValueError(
                "representation must be [B,D] or [B,M,D], got "
                f"{tuple(representation.shape)}."
            )

        if answer_proj.size(0) != representation.size(0):
            raise ValueError(
                "answer_proj and representation batch sizes differ."
            )
        if answer_proj.size(-1) != representation.size(-1):
            raise ValueError(
                "answer_proj and representation dimensions differ: "
                f"{answer_proj.size(-1)} vs "
                f"{representation.size(-1)}."
            )

        candidate_features = (
            answer_proj.detach()
            if detach_answer
            else answer_proj
        )
        if normalize:
            candidate_features = F.normalize(
                candidate_features,
                p=2,
                dim=-1,
                eps=1e-6,
            )
            representation = F.normalize(
                representation,
                p=2,
                dim=-1,
                eps=1e-6,
            )

        if logit_scale <= 0.0:
            raise ValueError(
                "logit_scale must be positive."
            )

        logits = float(logit_scale) * torch.einsum(
            "bcd,bmd->bmc",
            candidate_features,
            representation,
        )

        prop_num = int(
            self.cfgs["model"].get("prop_num", 1)
        )
        if prop_num < 1:
            raise ValueError(
                "model.prop_num must be positive."
            )

        total_candidates = logits.size(-1)
        if total_candidates % prop_num != 0:
            raise ValueError(
                "Candidate dimension is not divisible by prop_num: "
                f"{total_candidates} vs {prop_num}."
            )

        logits = logits.reshape(
            logits.size(0),
            logits.size(1),
            prop_num,
            total_candidates // prop_num,
        ).mean(dim=2)

        if squeeze_dim:
            logits = logits.squeeze(1)
        return logits

    def _project_qbound_value_with_frozen_host(
        self,
        pooled_value,
    ):
        """
        Map raw CLIP VALUE evidence into CRA's language-comparison space while
        stopping gradients into the CRA projection itself.
        """
        if pooled_value.ndim != 2:
            raise ValueError("pooled_value must be [B,D].")

        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        decoder = getattr(model_core, "decoder", None)
        if decoder is None:
            raise AttributeError(
                "Q-Bound V2.8 grounding requires model.decoder."
            )

        projection = getattr(decoder, "video_proj", None)
        if projection is None:
            raise AttributeError(
                "RefineDecoder must expose video_proj for the frozen "
                "grounding metric adapter."
            )

        linear = None
        for module in projection.modules():
            if isinstance(module, nn.Linear):
                linear = module
                break

        if linear is None:
            raise TypeError(
                "decoder.video_proj must contain an nn.Linear layer."
            )

        if pooled_value.size(-1) != linear.in_features:
            raise ValueError(
                "Q-Bound pooled VALUE dimension does not match host "
                f"projection input: {pooled_value.size(-1)} vs "
                f"{linear.in_features}."
            )

        weight = linear.weight.detach()
        bias = (
            linear.bias.detach()
            if linear.bias is not None
            else None
        )

        return F.linear(
            pooled_value,
            weight,
            bias,
        )

    def _qbound_video_question_objective(
        self,
        model_outputs,
        qbound_aux,
        temperature,
    ):
        """
        One-directional in-batch Video->Question semantic grounding.

        The video side is raw CLIP VALUE pooled by the differentiable V2.8
        Support Field.  It is mapped with detached host-projector weights.
        The target is QUESTION-ONLY and detached.  Since the support graph is
        detached upstream, this loss can train semantic relevance but cannot
        train the video-only support predictor or move the question encoder.
        """
        if not isinstance(model_outputs, dict):
            raise TypeError("model_outputs must be a dict.")
        if not isinstance(qbound_aux, dict):
            raise TypeError("qbound_aux must be a dict.")

        required = {
            "qbound_grounding_pooled_value",
            "qbound_grounding_question",
            "qbound_grounding_key",
        }
        missing = sorted(required.difference(model_outputs))
        if missing:
            raise KeyError(
                "Q-Bound V2.8 grounding interface is missing: "
                + ", ".join(missing)
            )

        pooled_value = model_outputs["qbound_grounding_pooled_value"]
        question_target = model_outputs["qbound_grounding_question"]
        grounding_key = model_outputs["qbound_grounding_key"]

        if pooled_value.ndim != 2 or question_target.ndim != 2:
            raise ValueError(
                "Grounded VALUE and question target must both be [B,D]."
            )

        metric_value = self._project_qbound_value_with_frozen_host(
            pooled_value
        )
        if metric_value.shape != question_target.shape:
            raise ValueError(
                "Frozen host-projected VALUE and question target must have "
                f"the same [B,D]; got {tuple(metric_value.shape)} and "
                f"{tuple(question_target.shape)}."
            )
        if grounding_key.ndim != 2 or grounding_key.size(0) != pooled_value.size(0):
            raise ValueError("qbound_grounding_key must be [B,T].")
        if question_target.requires_grad:
            raise RuntimeError(
                "V2.8 grounding question target must be stop-gradient."
            )
        if temperature <= 0.0:
            raise ValueError("grounding temperature must be positive.")

        batch_size = pooled_value.size(0)
        if batch_size < 2:
            zero = pooled_value.sum() * 0.0
            return zero, {
                "grounding_batch_size": float(batch_size),
                "grounding_retrieval_acc": 0.0,
                "grounding_positive_similarity": 0.0,
                "grounding_negative_similarity": 0.0,
                "grounding_hardest_negative_similarity": 0.0,
                "grounding_margin": 0.0,
            }

        video_unit = F.normalize(metric_value, p=2, dim=-1, eps=1e-6)
        question_unit = F.normalize(
            question_target, p=2, dim=-1, eps=1e-6
        )
        similarity = torch.matmul(
            video_unit,
            question_unit.transpose(0, 1),
        )
        logits = similarity / float(temperature)
        labels = torch.arange(
            batch_size,
            device=logits.device,
            dtype=torch.long,
        )
        grounding_loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            positive = similarity.diag()
            off_diagonal = ~torch.eye(
                batch_size,
                dtype=torch.bool,
                device=similarity.device,
            )
            negative = similarity[off_diagonal].reshape(
                batch_size, batch_size - 1
            )
            hardest_negative = negative.max(dim=-1).values
            predicted = logits.argmax(dim=-1)
            diagnostics = {
                "grounding_batch_size": float(batch_size),
                "grounding_retrieval_acc": float(
                    (predicted == labels).float().mean().item()
                ),
                "grounding_positive_similarity": float(
                    positive.mean().item()
                ),
                "grounding_negative_similarity": float(
                    negative.mean().item()
                ),
                "grounding_hardest_negative_similarity": float(
                    hardest_negative.mean().item()
                ),
                "grounding_margin": float(
                    (positive - hardest_negative).mean().item()
                ),
            }

        return grounding_loss, diagnostics

    # ------------------------------------------------------------------
    # V3.5 / QDTG — opt-in SAME-VIDEO question discrimination.
    # ------------------------------------------------------------------
    def _v35_pair_loss_settings(self):
        """Strictly gated so an unchanged R2-C YAML runs IDENTICAL code."""
        q_cfg = self._qbound_config() if self._qbound_enabled() else {}
        branch = q_cfg.get("pair_loss", {})
        if branch is None:
            branch = {}
        if not isinstance(branch, dict):
            raise TypeError("model.qbound.pair_loss must be a mapping/dict.")
        enabled = branch.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("model.qbound.pair_loss.enabled must be boolean.")
        weight = float(branch.get("loss_weight", 0.0))
        temperature = float(branch.get("temperature", 0.07))
        margin = float(branch.get("margin", 0.0))
        gradient_audit_every = branch.get("gradient_audit_every", 0)
        if not isinstance(gradient_audit_every, int) or isinstance(gradient_audit_every, bool):
            raise TypeError("model.qbound.pair_loss.gradient_audit_every must be an integer.")
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("V3.5 pair loss_weight must be finite and >= 0.")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("V3.5 pair temperature must be positive and finite.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("V3.5 pair margin must be finite and >= 0.")
        if gradient_audit_every < 0:
            raise ValueError("V3.5 pair gradient_audit_every must be >= 0.")
        if enabled != (weight > 0.0):
            raise ValueError(
                "V3.5 requires pair_loss.enabled=true with positive loss_weight "
                "OR enabled=false with loss_weight=0."
            )
        if not enabled and gradient_audit_every != 0:
            raise ValueError("V3.5 gradient audit requires enabled pair loss.")
        return dict(enabled=enabled, weight=weight, temperature=temperature,
                    margin=margin, gradient_audit_every=gradient_audit_every)

    @staticmethod
    def _v35_pair_inputs(batch, batch_size, device):
        """Fail closed: Q2 must be a separate question from the same actual video.

        Required TRAIN-only fields emitted by the future dataset/collate:
          paired_question: LongTensor[B,L2] (question-only ids, PAD-aware)
          pair_valid_mask: BoolTensor[B]
          paired_video_id: list[str] length B (source video of each Q2)
          paired_question_id: list[str] length B (source unique Q2 id)
        Original batch must include video_id and question_id.  Pair validity
        needs to be independently established by the DATASET, not inferred from
        answers.  Identical-video metadata alone cannot prove distinct GT spans.
        """
        required = ("paired_question", "pair_valid_mask", "paired_video_id",
                    "paired_question_id", "video_id", "question_id")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "V3.5 pair_loss is enabled but training batch is missing "
                f"{missing}. Update dataset.py/dataloaders.py before training."
            )
        q2 = batch["paired_question"]
        valid = batch["pair_valid_mask"]
        if not torch.is_tensor(q2) or q2.ndim != 2 or q2.size(0) != batch_size:
            raise ValueError("V3.5 paired_question must be tensor [B,L2].")
        if q2.dtype not in (torch.int32, torch.int64) or q2.size(1) < 1:
            raise TypeError("V3.5 paired_question must be nonempty integer token ids.")
        if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (batch_size,):
            raise TypeError("V3.5 pair_valid_mask must be BoolTensor[B].")

        def _metadata(name):
            value = batch[name]
            if not isinstance(value, (list, tuple)) or len(value) != batch_size:
                raise TypeError(f"V3.5 {name} must be a list/tuple of B identifiers.")
            return [str(x) for x in value]

        vid1 = _metadata("video_id")
        vid2 = _metadata("paired_video_id")
        qid1 = _metadata("question_id")
        qid2 = _metadata("paired_question_id")
        valid_cpu = valid.detach().cpu().tolist()
        for i, is_valid in enumerate(valid_cpu):
            if is_valid and vid1[i] != vid2[i]:
                raise ValueError(f"V3.5 row {i}: Q1/Q2 belong to DIFFERENT videos.")
            if is_valid and qid1[i] == qid2[i]:
                raise ValueError(f"V3.5 row {i}: Q1/Q2 refer to the SAME question id.")
        return q2.to(device=device, non_blocking=True), valid.to(device=device, non_blocking=True)

    def _v35_pair_objective(self, model_outputs, settings):
        """Question-discriminative evidence loss (TRAIN ONLY), no new weights.

        Per-frame u_t = normalize(W_host @ raw_CLIP_t + b_host), with detached
        host projection weights and detached video features. Center per-video
        on VALID frames: r_t = u_t - mean_valid(u).  Then e_i=sum_t S_i(t) r_t.
        NOTE: because sum(S1)=sum(S2)=1, centering cancels in (e1-e2).
        Thus centering is numerically redundant for D, NOT independent
        protection against semantic shortcuts or wrong-event localization.

        D = ((e_1-e_2) dot (q_1-q_2))/temperature, with stopgrad normalized
        question-only embeddings. L_pair = mean_valid softplus(margin-D).

        The loss DOES NOT identify GT event; its gradient must be validated
        before a full training run. A wrong Q-specific event is still possible.
        """
        if not self.model.training or not torch.is_grad_enabled():
            raise RuntimeError("V3.5 QDTG pair objective is strictly train-only.")
        required = (
            "qbound_support_temporal_prob", "qbound_pair_support_temporal_prob",
            "qbound_grounding_question", "qbound_pair_grounding_question",
            "qbound_pair_valid_mask", "qbound_value_sequence", "qbound_semantic_seed_logits",
            "qbound_pair_semantic_seed_logits", "qbound_video_mask",
        )
        missing = [key for key in required if key not in model_outputs]
        if missing:
            raise KeyError(f"V3.5 CRA outputs missing pair fields: {missing}")

        s1 = model_outputs["qbound_support_temporal_prob"]
        s2 = model_outputs["qbound_pair_support_temporal_prob"]
        q1 = model_outputs["qbound_grounding_question"]
        q2 = model_outputs["qbound_pair_grounding_question"]
        valid_pairs = model_outputs["qbound_pair_valid_mask"]
        raw = model_outputs["qbound_value_sequence"]
        frame_mask = model_outputs["qbound_video_mask"].bool()

        if s1.ndim != 2 or s1.shape != s2.shape or not s1.requires_grad or not s2.requires_grad:
            raise RuntimeError("V3.5 S1/S2 must be differentiable matching [B,T] Support Fields.")
        b, t = s1.shape
        if raw.ndim != 3 or raw.shape[:2] != (b, t):
            raise ValueError("V3.5 raw VALUE must be [B,T,D].")
        if q1.ndim != 2 or q1.shape != q2.shape or q1.size(0) != b:
            raise ValueError("V3.5 question-global representations must match [B,D].")
        if valid_pairs.shape != (b,) or valid_pairs.dtype != torch.bool:
            raise TypeError("V3.5 pair_valid_mask must be BoolTensor[B].")
        if frame_mask.shape != (b, t):
            raise ValueError("V3.5 qbound_video_mask must be [B,T].")
        if q1.requires_grad or q2.requires_grad:
            raise RuntimeError("V3.5 question targets must be detached.")
        if not (torch.isfinite(s1).all() and torch.isfinite(s2).all()
                and torch.isfinite(q1).all() and torch.isfinite(q2).all()):
            raise RuntimeError("V3.5 input contains non-finite values.")

        # Calculate in float32 for AMP safety.  The sum over valid frames is 1.
        # F.linear on raw features keeps video/projector out of THIS loss graph.
        source = raw.detach().float()
        host = self._v35_host_projection_linear()
        projected = F.linear(source, host.weight.detach().float(),
                             host.bias.detach().float() if host.bias is not None else None)
        u = F.normalize(projected, p=2, dim=-1, eps=1e-6)
        valid_float = frame_mask.unsqueeze(-1).to(dtype=u.dtype)
        mean = (u * valid_float).sum(dim=1, keepdim=True) / valid_float.sum(
            dim=1, keepdim=True).clamp_min(1.0)
        residual = (u - mean) * valid_float
        if residual.size(-1) != q1.size(-1):
            raise ValueError("V3.5 host-projected visual and question dimensions differ.")

        e1 = torch.einsum("bt,btd->bd", s1.float(), residual)
        e2 = torch.einsum("bt,btd->bd", s2.float(), residual)
        question_diff = F.normalize(q1.detach().float(), p=2, dim=-1, eps=1e-6) - F.normalize(
            q2.detach().float(), p=2, dim=-1, eps=1e-6)
        discrimination = ((e1 - e2) * question_diff).sum(dim=-1) / settings["temperature"]
        per_pair = F.softplus(settings["margin"] - discrimination)

        if valid_pairs.any():
            pair_loss = per_pair[valid_pairs].mean()
        else:
            # Zero gradient, no made-up pair. The dataset controls coverage.
            pair_loss = s1.sum() * 0.0

        with torch.no_grad():
            v = valid_pairs
            n_valid = int(v.sum().item())
            info = {
                "qbound_pair_weight": float(settings["weight"]),
                "qbound_pair_temperature": float(settings["temperature"]),
                "qbound_pair_margin": float(settings["margin"]),
                "qbound_pair_valid_fraction": float(v.float().mean().item()),
                "qbound_pair_valid_count": float(n_valid),
                "qbound_pair_discrimination": float(discrimination[v].mean().item()) if n_valid else 0.0,
                "qbound_pair_margin_positive_fraction": float((discrimination[v] > 0).float().mean().item()) if n_valid else 0.0,
                "qbound_pair_support_l1": float((s1[v] - s2[v]).abs().sum(dim=-1).mean().item()) if n_valid else 0.0,
                "qbound_pair_question_cosine": float(F.cosine_similarity(q1[v].float(), q2[v].float(), dim=-1).mean().item()) if n_valid else 0.0,
                "qbound_pair_visual_residual_norm": float(residual[frame_mask].norm(dim=-1).mean().item()) if frame_mask.any() else 0.0,
            }
        if not torch.isfinite(pair_loss).all():
            raise RuntimeError("V3.5 L_pair non-finite.")
        return pair_loss, info

    def _v35_host_projection_linear(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "video_proj"):
            raise AttributeError("V3.5 requires CRA RefineDecoder.video_proj.")
        layers = [x for x in decoder.video_proj.modules() if isinstance(x, nn.Linear)]
        if len(layers) != 1:
            raise RuntimeError("V3.5 expects exactly one nn.Linear in video_proj.")
        return layers[0]

    @staticmethod
    def _v35_pair_gradient_diagnostics(pair_loss, grounding_loss, model_outputs,
                                       pair_weight, grounding_weight):
        """Measure loss-gradient agreement at Q1 Seed logits; NOT GT direction."""
        seed1 = model_outputs["qbound_semantic_seed_logits"]
        seed2 = model_outputs["qbound_pair_semantic_seed_logits"]
        if not seed1.requires_grad or not seed2.requires_grad:
            raise RuntimeError("V3.5 Seed logits must retain gradients.")
        gp1, gp2 = torch.autograd.grad(pair_loss, (seed1, seed2),
                                      retain_graph=True, allow_unused=False)
        (gg1,) = torch.autograd.grad(grounding_loss, (seed1,),
                                    retain_graph=True, allow_unused=False)
        p = gp1.detach().float().reshape(gp1.size(0), -1)
        g = gg1.detach().float().reshape(gg1.size(0), -1)
        p_norm, g_norm = p.norm(dim=-1), g.norm(dim=-1)
        cosine = (p*g).sum(dim=-1) / (p_norm*g_norm).clamp_min(1e-12)
        weighted_ratio = (float(pair_weight) * p.norm()) / (
            float(grounding_weight) * g.norm() + 1e-12)
        return {
            "qbound_pair_grad_weighted_ratio": float(weighted_ratio.item()),
            "qbound_pair_grad_seed1_norm": float(p.norm().item()),
            "qbound_pair_grad_seed2_norm": float(gp2.detach().float().norm().item()),
            "qbound_pair_grad_grounding_norm": float(g.norm().item()),
            "qbound_pair_grad_cosine_grounding": float(cosine.mean().item()),
            "qbound_pair_grad_conflict_fraction": float((cosine < 0).float().mean().item()),
        }

    def _validate_v280_contract(self, gqa_output):
        """Validate the pipeline-facing V2.8 causal and gradient contract."""
        if gqa_output is None or len(gqa_output) < 3:
            raise TypeError("Q-Bound V2.8 expects a 3-item CRA output.")

        main_video_proj = gqa_output[0]
        model_outputs = gqa_output[2]
        if not isinstance(model_outputs, dict):
            raise TypeError("CRA model outputs must be a dict.")

        qbound_aux = self._get_qbound_aux(gqa_output)
        if not isinstance(qbound_aux, dict):
            raise KeyError(
                "Q-Bound V2.8 must expose time_param['qbound_aux']."
            )
        self._h1_check_runtime(gqa_output)
        self._v37_check_runtime(gqa_output)

        required_aux = {
            "qbound_version",
            "video_mask",
            "semantic_seed_prob",
            "support_temporal_prob",
            "support_edge_mask",
            "support_edge_affinity",
            "support_graph_laplacian",
            "support_predictive_loss",
            "qbound_support_predictive_loss",
            "segment_prob",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
            "qbound_qa_video_proj",
            "qbound_grounding_video_proj",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
            "cra_key",
            "structured_visual_evidence_enabled",
            "relevance_extent_decoupling_enabled",
            "token_resolved_question_relevance_enabled",
            "semantic_evidence_seed_enabled",
            "video_only_support_branch_enabled",
            "predictive_support_graph_enabled",
            "support_graph_detached_for_grounding_enabled",
            "structural_seed_propagation_enabled",
            "analytic_boundary_posterior_enabled",
            "posterior_barycentric_grounding_enabled",
            "same_support_for_training_and_inference_enabled",
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_predictive_loss_exposed",
            "visual_enrichment_only_affects_where",
            "question_conditions_relevance_only",
            "video_only_extent_structure",
            "shared_support_for_qa_and_grounding",
            "qa_support_detached",
            "posterior_barycentric_span_is_metadata_only",
            "question_grounding_objective_expected",
            "support_predictive_objective_expected",
            "grounding_question_stop_gradient",
            "question_direct_span_scoring_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "hard_map_temporal_decision_enabled",
            "neural_span_scorer_enabled",
            "event_candidate_selector_enabled",
            "event_relative_enabled_flag",
            "answer_features_used_in_core",
            "temporal_gt_used_in_core",
            "answer_candidates_used_in_qbound_temporal_path",
            "vqa_gradient_to_qbound_topology_enabled",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.8 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_outputs = {
            "qbound_grounding_question",
            "qbound_grounding_pooled_value",
            "qbound_grounding_video_proj",
            "qbound_grounding_key",
            "qbound_vqa_video_proj",
            "qbound_vqa_key",
            "qbound_support_predictive_loss",
        }
        missing_outputs = sorted(required_outputs.difference(model_outputs))
        if missing_outputs:
            raise KeyError(
                "CRA/Q-Bound V2.8 integration output is missing: "
                + ", ".join(missing_outputs)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRAPipeline/CRA/Q-Bound version mismatch: expected "
                f"{self.qbound_version!r}, got "
                f"{qbound_aux['qbound_version']!r}."
            )

        def _flag(name):
            value = qbound_aux[name]
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"{name} must be a scalar flag.")
                value = value.detach().float().item()
            return float(value)

        expected_on = {
            "structured_visual_evidence_enabled",
            "relevance_extent_decoupling_enabled",
            "token_resolved_question_relevance_enabled",
            "semantic_evidence_seed_enabled",
            "video_only_support_branch_enabled",
            "predictive_support_graph_enabled",
            "support_graph_detached_for_grounding_enabled",
            "structural_seed_propagation_enabled",
            "analytic_boundary_posterior_enabled",
            "posterior_barycentric_grounding_enabled",
            "same_support_for_training_and_inference_enabled",
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_predictive_loss_exposed",
            "visual_enrichment_only_affects_where",
            "question_conditions_relevance_only",
            "video_only_extent_structure",
            "shared_support_for_qa_and_grounding",
            "qa_support_detached",
            "posterior_barycentric_span_is_metadata_only",
            "question_grounding_objective_expected",
            "support_predictive_objective_expected",
            "grounding_question_stop_gradient",
        }
        expected_off = {
            "question_direct_span_scoring_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "hard_map_temporal_decision_enabled",
            "neural_span_scorer_enabled",
            "event_candidate_selector_enabled",
            "answer_features_used_in_core",
            "temporal_gt_used_in_core",
            "answer_candidates_used_in_qbound_temporal_path",
            "vqa_gradient_to_qbound_topology_enabled",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
        }
        if not self.r2_predictor_enabled:
            predictive_flags = {"video_only_support_branch_enabled",
                                "predictive_support_graph_enabled", "qbound_predictive_loss_exposed",
                                "support_predictive_objective_expected"}
            expected_on -= predictive_flags
            expected_off |= predictive_flags
        if self.v37_event_mode == "event_relative":
            # These three flags describe actual V3.7 candidate scoring.
            # They cannot be forced to 0 to bypass old V2.8 assertions.
            event_flags = {
                "question_direct_span_scoring_enabled",
                "neural_span_scorer_enabled",
                "event_candidate_selector_enabled",
            }
            expected_off -= event_flags
            expected_on |= event_flags
        for name in expected_on:
            actual = _flag(name)
            if actual != 1.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 1."
                )
        for name in expected_off:
            actual = _flag(name)
            if actual != 0.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 0."
                )

        def _finite(name, value):
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor.")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{name} contains non-finite values.")

        video_mask = qbound_aux["video_mask"].bool()
        if video_mask.ndim != 2 or not torch.all(video_mask.any(dim=-1)):
            raise ValueError("video_mask must be [B,T] with >=1 valid frame.")
        batch_size, time_steps = video_mask.shape

        def _distribution(name, value):
            if value.shape != video_mask.shape:
                raise ValueError(
                    f"{name} must be [B,T]={tuple(video_mask.shape)}, got "
                    f"{tuple(value.shape)}."
                )
            _finite(name, value)
            if torch.any(value.detach() < -1e-6):
                raise RuntimeError(f"{name} contains negative mass.")
            invalid_mass = (
                value.detach() * (~video_mask).to(value.dtype)
            ).abs().amax()
            if float(invalid_mass.item()) > 1e-5:
                raise RuntimeError(f"{name} assigns mass to padding.")
            total = (
                value.detach() * video_mask.to(value.dtype)
            ).sum(dim=-1)
            if not torch.allclose(
                total,
                torch.ones_like(total),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise RuntimeError(f"{name} must sum to one.")

        distribution_names = (
            "semantic_seed_prob",
            "support_temporal_prob",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
        )
        for name in distribution_names:
            _distribution(name, qbound_aux[name])

        support = qbound_aux["support_temporal_prob"].detach()
        for name in distribution_names[2:]:
            if not torch.allclose(
                qbound_aux[name].detach(),
                support,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    f"{name} diverges from the single V2.8 Support Field."
                )

        for name in ("qbound_qa_key", "final_key"):
            if qbound_aux[name].requires_grad:
                raise RuntimeError(
                    f"{name} must detach support before the CRA/QA route."
                )

        training_with_grad = self.model.training and torch.is_grad_enabled()
        if training_with_grad:
            for name in ("semantic_seed_prob", "qbound_grounding_key"):
                if not qbound_aux[name].requires_grad:
                    raise RuntimeError(
                        f"{name} must remain differentiable for L_ground."
                    )

        edge_steps = max(time_steps - 1, 0)
        edge_mask = qbound_aux["support_edge_mask"].bool()
        if edge_mask.shape != (batch_size, edge_steps):
            raise ValueError("support_edge_mask has the wrong [B,T-1] shape.")
        edge_affinity = qbound_aux["support_edge_affinity"]
        if edge_affinity.shape != edge_mask.shape:
            raise ValueError("support_edge_affinity must match edge_mask.")
        _finite("support_edge_affinity", edge_affinity)
        if edge_affinity.requires_grad:
            raise RuntimeError(
                "L_ground must see a detached support_edge_affinity."
            )
        # Mode contract at pipeline level, in addition to Refine/Base checks.
        # It must remain true in both training and validation inference.
        actual_mode = qbound_aux.get("support_graph_mode", "learned")
        if actual_mode != self.r2_graph_mode:
            raise RuntimeError(
                "R2 graph mode mismatch in VQA outputs: "
                f"config={self.r2_graph_mode!r}, aux={actual_mode!r}."
            )
        expected_used = (
            edge_mask.to(edge_affinity.dtype)
            if actual_mode == "fixed"
            else qbound_aux["support_edge_affinity_raw"].detach()
        )
        if not torch.allclose(
            edge_affinity.detach(), expected_used.detach(), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError(
                f"R2 {actual_mode} propagation received unexpected edges."
            )

        laplacian = qbound_aux["support_graph_laplacian"]
        if laplacian.shape != (batch_size, time_steps, time_steps):
            raise ValueError("support_graph_laplacian must be [B,T,T].")
        _finite("support_graph_laplacian", laplacian)
        if laplacian.requires_grad:
            raise RuntimeError(
                "L_ground must see a detached support graph Laplacian."
            )

        predictive_loss = qbound_aux["support_predictive_loss"]
        if not torch.is_tensor(predictive_loss) or predictive_loss.numel() != 1:
            raise TypeError("support_predictive_loss must be a scalar tensor.")
        _finite("support_predictive_loss", predictive_loss)
        if self.r2_predictor_enabled and training_with_grad and bool(edge_mask.any()) and not predictive_loss.requires_grad:
            raise RuntimeError(
                "L_pred is detached; the support predictor would not learn."
            )
        if not self.r2_predictor_enabled and (
            predictive_loss.requires_grad or float(predictive_loss.detach().item()) != 0.0
        ):
            raise RuntimeError("R2-C must not compute L_pred.")
        for alias in (
            qbound_aux["qbound_support_predictive_loss"],
            model_outputs["qbound_support_predictive_loss"],
        ):
            if not torch.allclose(
                alias.detach(),
                predictive_loss.detach(),
                atol=1e-7,
                rtol=1e-7,
            ):
                raise RuntimeError("Predictive-loss aliases disagree.")

        segment_prob = qbound_aux["segment_prob"]
        if segment_prob.shape != (batch_size, time_steps, time_steps):
            raise ValueError("segment_prob must be [B,T,T].")
        _finite("segment_prob", segment_prob)
        segment_total = segment_prob.detach().sum(dim=(-2, -1))
        if not torch.allclose(
            segment_total,
            torch.ones_like(segment_total),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError("Analytic V2.8 segment_prob must sum to one.")

        qa_feature = qbound_aux["qbound_qa_video_proj"]
        grounding_feature = qbound_aux["qbound_grounding_video_proj"]
        for name, value in (
            ("main_video_proj", main_video_proj),
            ("qbound_qa_video_proj", qa_feature),
            ("qbound_grounding_video_proj", grounding_feature),
            ("model_outputs.qbound_vqa_video_proj", model_outputs["qbound_vqa_video_proj"]),
        ):
            _finite(name, value)
            if value.shape != main_video_proj.shape:
                raise ValueError(f"{name} does not match the CRA QA feature.")
        if not torch.allclose(
            main_video_proj.detach(), qa_feature.detach(), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("CRA must return the detached-support QA view.")
        if not torch.allclose(
            qa_feature.detach(), grounding_feature.detach(), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError(
                "QA and grounding must use the same numerical Support Field."
            )

        for name in (
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        ):
            value = qbound_aux[name]
            _finite(name, value)
            if float(value.detach().amax().item()) > 1e-5:
                raise RuntimeError(f"{name} violates the shared-support contract.")

        question_target = model_outputs["qbound_grounding_question"]
        _finite("qbound_grounding_question", question_target)
        if question_target.requires_grad:
            raise RuntimeError(
                "The V2.8 question-only grounding target must be detached."
            )

        return qbound_aux

    def _collect_qbound_stats(
        self,
        gqa_output,
        extra_logs=None,
    ):
        stats = {}
        if extra_logs:
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return stats

        scalar_keys = (
            "visual_distinctiveness_input_ratio",
            "visual_distinctiveness_delta_ratio",
            "visual_transition_delta_ratio",
            "visual_enrichment_delta_ratio",
            "temporal_context_delta_ratio",
            # H1 module capacity/use: ratios are not GT correctness metrics.
            "eqptc_early_delta_ratio",
            "eqptc_temporal_delta_ratio",
            "eqptc_enabled_flag",
            "eqptc_early_scale_used",
            "eqptc_position_scale_used",
            "eqptc_temporal_scale_used",
            # V3.7 event diagnostics (all computed without temporal GT).
            "event_relative_enabled_flag",
            "event_residual_scale_used",
            "event_relation_temperature_used",
            "event_candidate_count",
            "event_candidate_top1_prob",
            "event_residual_rms",
            "event_residual_to_legacy_std_ratio",
            "relevance_question_attention_entropy_ratio",
            "semantic_seed_entropy_ratio",
            "semantic_seed_top1_prob",
            "semantic_seed_effective_frames",
            "support_edge_surprise_mean",
            "support_edge_surprise_std",
            "support_edge_affinity_mean",
            "support_edge_affinity_std",
            "support_predictive_loss_per_sample",
            "support_predictive_loss",
            "support_entropy_ratio",
            "support_top1_prob",
            "support_effective_frames",
            "support_propagation_l1_delta",
            "semantic_seed_support_cosine",
            "start_boundary_entropy_ratio",
            "end_boundary_entropy_ratio",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_single_frame_indicator",
            "selected_full_video_indicator",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        )
        for key in scalar_keys:
            if not getattr(self, "r2_predictor_enabled", True) and (
                key.startswith("support_predictive")
                or key.startswith("support_edge_surprise")
            ):
                continue
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                log_key = key if key.startswith("qbound_") else f"qbound_{key}"
                stats[log_key] = scalar

        edge_mask = qbound_aux.get("support_edge_mask")
        edge_names = (
            ("support_edge_surprise", "support_edge_affinity_raw", "support_edge_affinity")
            if getattr(self, "r2_predictor_enabled", True)
            else ("support_edge_affinity",)
        )
        for name in edge_names:
            value = qbound_aux.get(name)
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            detached = value.detach().float()
            if torch.is_tensor(edge_mask) and edge_mask.shape == detached.shape:
                detached = detached[edge_mask.detach().bool()]
            if detached.numel() == 0:
                continue
            stats[f"qbound_{name}_mean"] = detached.mean().item()
            stats[f"qbound_{name}_std"] = detached.std(unbiased=False).item()
            stats[f"qbound_{name}_min"] = detached.min().item()
            stats[f"qbound_{name}_max"] = detached.max().item()

        video_mask = qbound_aux.get("video_mask")
        mask = video_mask.detach().bool() if torch.is_tensor(video_mask) else None
        seed_key = qbound_aux.get("semantic_seed_prob")
        support_key = qbound_aux.get("support_temporal_prob")
        qa_key = qbound_aux.get("qbound_qa_key")
        grounding_key = qbound_aux.get("qbound_grounding_key")
        cra_key = qbound_aux.get("cra_key")

        def _compare(prefix, first, second):
            if (
                not torch.is_tensor(first)
                or not torch.is_tensor(second)
                or first.shape != second.shape
                or first.ndim != 2
            ):
                return
            first = self._normalize_distribution(first.detach(), mask=mask)
            second = self._normalize_distribution(second.detach(), mask=mask)
            stats[f"qbound_{prefix}_js"] = self._js_divergence(
                first, second, mask=mask
            ).mean().item()
            stats[f"qbound_{prefix}_l1"] = (
                (first - second).abs().sum(dim=-1).mean().item()
            )

        _compare("seed_support", seed_key, support_key)
        if self.v37_event_mode == "event_relative":
            legacy_seed = qbound_aux.get("legacy_semantic_seed_prob")
            event_frame = qbound_aux.get("event_frame_prob")
            _compare("event_legacy_fused_seed", legacy_seed, seed_key)
            _compare("event_frame_fused_seed", event_frame, seed_key)
            _compare("event_frame_support", event_frame, support_key)
            if torch.is_tensor(legacy_seed) and torch.is_tensor(seed_key):
                stats["qbound_event_legacy_fused_peak_change_fraction"] = float(
                    (legacy_seed.detach().argmax(dim=-1) !=
                     seed_key.detach().argmax(dim=-1)).float().mean().item()
                )
            candidate = qbound_aux.get("event_candidate_prob")
            if torch.is_tensor(candidate):
                candidate = candidate.detach().float().clamp_min(1e-9)
                n_valid = qbound_aux["event_candidate_valid"].sum(-1).float().clamp_min(1.0)
                ent = -(candidate * candidate.log()).sum(-1)
                stats["qbound_event_candidate_entropy_ratio"] = float(
                    (ent / n_valid.log().clamp_min(1.0)).mean().item()
                )
        _compare("cra_support", cra_key, support_key)
        _compare("qa_grounding_support", qa_key, grounding_key)

        for new_name, support_name, seed_name in (
            (
                "qbound_support_minus_seed_entropy_ratio",
                "support_entropy_ratio",
                "semantic_seed_entropy_ratio",
            ),
            (
                "qbound_support_minus_seed_effective_frames",
                "support_effective_frames",
                "semantic_seed_effective_frames",
            ),
            (
                "qbound_support_minus_seed_top1_prob",
                "support_top1_prob",
                "semantic_seed_top1_prob",
            ),
        ):
            support_value = qbound_aux.get(support_name)
            seed_value = qbound_aux.get(seed_name)
            if torch.is_tensor(support_value) and torch.is_tensor(seed_value):
                stats[new_name] = (
                    support_value.detach().float()
                    - seed_value.detach().float()
                ).mean().item()

        return stats

    def _qbound_grad_norm(self):
        if not self._qbound_enabled():
            return None

        total_sq = 0.0
        found = False
        for name, parameter in (
            self.model.named_parameters()
        ):
            if (
                "qbound" not in name.lower()
                or parameter.grad is None
            ):
                continue

            grad = (
                parameter.grad
                .detach()
                .float()
            )
            total_sq += (
                grad.pow(2).sum().item()
            )
            found = True

        return (
            float(total_sq ** 0.5)
            if found
            else 0.0
        )

    def _qbound_component_grad_norms(self):
        names = (
            "where_input", "visual_distinctiveness", "visual_transition",
            "temporal_context", "semantic_relevance",
        ) + (("eqptc_early", "eqptc_temporal")
             if getattr(self, "h1_temporal_mode", None) == "eq_ptc" else ()) \
          + (("event_comparator",) if getattr(self, "v37_event_mode", "legacy") == "event_relative" else ()) \
          + (("support_predictor",) if getattr(self, "r2_predictor_enabled", True) else ())
        totals = {name: 0.0 for name in names}
        if not self._qbound_enabled():
            return {
                f"qbound_{name}_grad_norm": 0.0
                for name in names
            }

        groups = {
            "where_input": ("where_input_proj",),
            "visual_distinctiveness": (
                "visual_distinctiveness_encoder",
            ),
            "visual_transition": (
                "visual_transition_encoder",
                "visual_transition_frame_delta",
            ),
            "temporal_context": (
                "temporal_norm1",
                "temporal_attn",
                "temporal_attn_delta",
                "temporal_norm2",
                "temporal_ff",
            ),
            "semantic_relevance": (
                "question_token_proj",
                "relevance_visual_norm",
                "relevance_question_attn",
                "relevance_question_context_norm",
                "semantic_seed_encoder",
                "semantic_seed_score",
            ),
            "eqptc_early": (
                "eqptc_early_visual_norm",
                "eqptc_early_question_attn",
                "eqptc_early_delta",
            ),
            "eqptc_temporal": (
                "eqptc_temporal_norm",
                "eqptc_temporal_attn",
                "eqptc_temporal_delta",
            ),
            "event_comparator": (
                "event_comparator.",
            ),
            "support_predictor": (
                "support_input_proj",
                "support_forward_cell",
                "support_backward_cell",
                "support_forward_predictor",
                "support_backward_predictor",
            ),
        }

        for parameter_name, parameter in self.model.named_parameters():
            if parameter.grad is None or "qbound" not in parameter_name.lower():
                continue
            lower = parameter_name.lower()
            grad_sq = (
                parameter.grad.detach().float().pow(2).sum().item()
            )
            for group_name, tokens in groups.items():
                if any(token in lower for token in tokens):
                    totals[group_name] += grad_sq
                    break

        result = {
            f"qbound_{name}_grad_norm": float(totals[name] ** 0.5)
            for name in names
        }
        relevance_sq = sum(
            totals[name]
            for name in (
                "where_input",
                "visual_distinctiveness",
                "visual_transition",
                "temporal_context",
                "semantic_relevance",
            ) + (("eqptc_early", "eqptc_temporal")
                 if getattr(self, "h1_temporal_mode", None) == "eq_ptc" else ())
            + (("event_comparator",) if getattr(self, "v37_event_mode", "legacy") == "event_relative" else ())
        )
        result["qbound_relevance_path_grad_norm"] = float(
            relevance_sq ** 0.5
        )
        return result

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_where_input_grad_norm",
            "qbound_visual_distinctiveness_grad_norm",
            "qbound_visual_transition_grad_norm",
            "qbound_temporal_context_grad_norm",
            "qbound_semantic_relevance_grad_norm",
            "qbound_eqptc_early_grad_norm",
            "qbound_eqptc_temporal_grad_norm",
            "qbound_event_comparator_grad_norm",
            "qbound_support_predictor_grad_norm",
            "qbound_relevance_path_grad_norm",
            "qbound_grounding_weight",
            "qbound_grounding_temperature",
            "qbound_predictive_weight",
            "qbound_grounding_loss",
            "qbound_grounding_weighted_loss",
            "qbound_predictive_loss",
            "qbound_predictive_weighted_loss",
            "qbound_grounding_batch_size",
            "qbound_grounding_retrieval_acc",
            "qbound_grounding_positive_similarity",
            "qbound_grounding_negative_similarity",
            "qbound_grounding_hardest_negative_similarity",
            "qbound_grounding_margin",
            # V3.5 / QDTG (populated only when pair_loss enabled)
            "qbound_pair_weight",
            "qbound_pair_temperature",
            "qbound_pair_margin",
            "qbound_pair_loss",
            "qbound_pair_weighted_loss",
            "qbound_pair_valid_fraction",
            "qbound_pair_valid_count",
            "qbound_pair_discrimination",
            "qbound_pair_margin_positive_fraction",
            "qbound_pair_support_l1",
            "qbound_pair_question_cosine",
            "qbound_pair_visual_residual_norm",
            "qbound_pair_grad_weighted_ratio",
            "qbound_pair_grad_seed1_norm",
            "qbound_pair_grad_seed2_norm",
            "qbound_pair_grad_grounding_norm",
            "qbound_pair_grad_cosine_grounding",
            "qbound_pair_grad_conflict_fraction",
            "qbound_visual_distinctiveness_input_ratio",
            "qbound_visual_distinctiveness_delta_ratio",
            "qbound_visual_transition_delta_ratio",
            "qbound_visual_enrichment_delta_ratio",
            "qbound_temporal_context_delta_ratio",
            "qbound_eqptc_early_delta_ratio",
            "qbound_eqptc_temporal_delta_ratio",
            "qbound_eqptc_enabled_flag",
            "qbound_eqptc_early_scale_used",
            "qbound_eqptc_position_scale_used",
            "qbound_eqptc_temporal_scale_used",
            "qbound_event_relative_enabled_flag",
            "qbound_event_residual_scale_used",
            "qbound_event_relation_temperature_used",
            "qbound_event_candidate_count",
            "qbound_event_candidate_top1_prob",
            "qbound_event_candidate_entropy_ratio",
            "qbound_event_residual_rms",
            "qbound_event_residual_to_legacy_std_ratio",
            "qbound_event_legacy_fused_seed_js",
            "qbound_event_legacy_fused_seed_l1",
            "qbound_event_frame_fused_seed_js",
            "qbound_event_frame_fused_seed_l1",
            "qbound_event_frame_support_js",
            "qbound_event_frame_support_l1",
            "qbound_event_legacy_fused_peak_change_fraction",
            "qbound_relevance_question_attention_entropy_ratio",
            "qbound_semantic_seed_entropy_ratio",
            "qbound_semantic_seed_top1_prob",
            "qbound_semantic_seed_effective_frames",
            "qbound_support_edge_surprise_mean",
            "qbound_support_edge_surprise_std",
            "qbound_support_edge_surprise_min",
            "qbound_support_edge_surprise_max",
            "qbound_support_edge_affinity_raw_mean",
            "qbound_support_edge_affinity_raw_std",
            "qbound_support_edge_affinity_raw_min",
            "qbound_support_edge_affinity_raw_max",
            "qbound_support_edge_affinity_mean",
            "qbound_support_edge_affinity_std",
            "qbound_support_edge_affinity_min",
            "qbound_support_edge_affinity_max",
            "qbound_support_predictive_loss_per_sample",
            "qbound_support_predictive_loss",
            "qbound_support_entropy_ratio",
            "qbound_support_top1_prob",
            "qbound_support_effective_frames",
            "qbound_support_propagation_l1_delta",
            "qbound_semantic_seed_support_cosine",
            "qbound_support_minus_seed_entropy_ratio",
            "qbound_support_minus_seed_effective_frames",
            "qbound_support_minus_seed_top1_prob",
            "qbound_start_boundary_entropy_ratio",
            "qbound_end_boundary_entropy_ratio",
            "qbound_segment_entropy_ratio",
            "qbound_segment_top1_prob",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_soft_expected_span_length",
            "qbound_soft_expected_span_fraction",
            "qbound_single_frame_prob_mass",
            "qbound_short_span_prob_mass",
            "qbound_selected_single_frame_indicator",
            "qbound_selected_full_video_indicator",
            "qbound_seed_support_js",
            "qbound_seed_support_l1",
            "qbound_cra_support_js",
            "qbound_cra_support_l1",
            "qbound_qa_grounding_support_js",
            "qbound_qa_grounding_support_l1",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        ]

    def get_loss(self, batch, qbound_mix=1.0):
        del qbound_mix
        device = self._model_device()

        answer_id = batch["answer_id"]
        answer = batch["answer"].to(device, non_blocking=True)
        video_frames = batch["video_frames"].to(
            device, non_blocking=True
        )
        question = batch["question"].to(
            device, non_blocking=True
        )
        qsns_id = batch["qsns_id"]
        qsns_token_ids = batch["qsns_token_ids"]
        video_len = batch["video_len"]

        settings = self._qbound_v280_loss_settings()
        video_mask = (
            get_mask(video_len, self.cfgs["dataset"]["max_feats"]).to(
                device, non_blocking=True
            )
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(answer_id, device)
        prediction_dict = {}
        target_dict = {}

        # V3.5 is train-only. Missing Q2 metadata must be a HARD ERROR, never
        # silently turned into zero pair loss over an unchanged R2-C run.
        v35_pair_kwargs = {}
        if self.v35_pair_settings["enabled"]:
            paired_question, pair_valid_mask = self._v35_pair_inputs(
                batch=batch, batch_size=batch_size, device=device)
            v35_pair_kwargs = {
                "paired_question": paired_question,
                "pair_valid_mask": pair_valid_mask,
            }

        # Historical CRA VQ objective remains outside Q-Bound.
        if self.cfgs["model"]["vg_loss"]:
            vq_output = self.model(
                video_frames,
                video_mask,
                answer=qsns_token_ids,
                answer_id=qsns_id,
                mode="VQ",
            )
            vq_video_proj = vq_output[0].unsqueeze(-1)
            vq_predicts = torch.bmm(
                vq_output[1], vq_video_proj
            ).squeeze(-1)
            vq_predicts = vq_predicts.reshape(
                batch_size,
                self.cfgs["model"]["prop_num"],
                -1,
            ).mean(dim=1)
            prediction_dict["vq_pred"] = vq_predicts
            target_dict["vq_target"] = qsns_id.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ).reshape(-1)
            prediction_dict["vq_align_pred"] = (
                vq_output[2]["time_param"]["grounding_feature"]
            )
            target_dict["vq_align_gt"] = vq_output[2]["align_gt"]

        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
            answer_only=None,
            **v35_pair_kwargs,
        )
        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]
        model_outputs = gqa_output[2]

        main_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )
        prediction_dict["vqa_pred"] = main_logits
        target_dict["vqa_target"] = answer_id_device

        # Critical V2.8 firewall: the existing CRA alignment objective belongs
        # to L_CRA and therefore consumes the detached-support QA view.  The
        # differentiable grounding view is reserved exclusively for L_ground.
        prediction_dict["vqa_align_pred"] = main_video_proj
        target_dict["vqa_align_gt"] = model_outputs["align_gt"]

        loss = self.criterion(prediction_dict, target_dict)
        if "total_loss" not in loss:
            raise KeyError("The criterion must return loss['total_loss'].")
        if not torch.isfinite(loss["total_loss"]).all():
            raise RuntimeError("L_CRA/main contains non-finite values.")

        qbound_logs = {}
        if self._qbound_enabled():
            qbound_aux = self._validate_v280_contract(gqa_output)

            grounding_loss, grounding_diag = (
                self._qbound_video_question_objective(
                    model_outputs=model_outputs,
                    qbound_aux=qbound_aux,
                    temperature=settings["grounding_temperature"],
                )
            )
            predictive_loss = qbound_aux["support_predictive_loss"]
            if not torch.isfinite(predictive_loss).all():
                raise RuntimeError("L_pred contains non-finite values.")

            weighted_grounding = (
                settings["grounding_weight"] * grounding_loss
            )
            loss["qbound_grounding_loss"] = grounding_loss
            loss["qbound_grounding_weighted_loss"] = weighted_grounding
            if self.r2_predictor_enabled:
                weighted_predictive = settings["predictive_weight"] * predictive_loss
                loss["qbound_predictive_loss"] = predictive_loss
                loss["qbound_predictive_weighted_loss"] = weighted_predictive
                loss["total_loss"] = loss["total_loss"] + weighted_grounding + weighted_predictive
            else:
                # No predictive loss calculation/gradient in R2-C.
                loss["total_loss"] = loss["total_loss"] + weighted_grounding

            # V3.5: only the opt-in pair objective can update the SAME shared
            # Q-Bound Seed scorer through Q1 and Q2 Support Fields.  Do not
            # change CRA QA, detached host/video/question sources or L_ground.
            if self.v35_pair_settings["enabled"]:
                pair_loss, pair_diag = self._v35_pair_objective(
                    model_outputs=model_outputs,
                    settings=self.v35_pair_settings,
                )
                weighted_pair = self.v35_pair_settings["weight"] * pair_loss
                loss["qbound_pair_loss"] = pair_loss
                loss["qbound_pair_weighted_loss"] = weighted_pair
                loss["total_loss"] = loss["total_loss"] + weighted_pair
                qbound_logs.update(pair_diag)
                self._v35_loss_step += 1
                period = self.v35_pair_settings["gradient_audit_every"]
                if period and self._v35_loss_step % period == 0 and pair_diag["qbound_pair_valid_count"] > 0:
                    qbound_logs.update(self._v35_pair_gradient_diagnostics(
                        pair_loss=pair_loss,
                        grounding_loss=grounding_loss,
                        model_outputs=model_outputs,
                        pair_weight=self.v35_pair_settings["weight"],
                        grounding_weight=settings["grounding_weight"],
                    ))

            qbound_logs.update(
                {
                    f"qbound_{name}": value
                    for name, value in grounding_diag.items()
                }
            )
            qbound_logs["qbound_grounding_weight"] = settings[
                "grounding_weight"
            ]
            qbound_logs["qbound_grounding_temperature"] = settings[
                "grounding_temperature"
            ]
            if self.r2_predictor_enabled:
                qbound_logs["qbound_predictive_weight"] = settings["predictive_weight"]

        predicted = main_logits.argmax(dim=-1).detach().cpu()
        running_accuracy = (
            predicted == answer_id.reshape(-1).cpu()
        ).float().mean().item()
        qbound_stats = self._collect_qbound_stats(
            gqa_output,
            extra_logs=qbound_logs,
        )
        return loss, running_accuracy, qbound_stats

    @staticmethod
    def _r1_posthoc_frame0_fixed(vatt, cur, thd_weight=0.3, dis=10):
        """R1 diagnostic: original find_seg_ada with only left bound >0 -> >=0.

        This is NOT used for primary R2 model selection or the canonical metrics.
        Its threshold, time discount, gap-skipping and right search are unchanged.
        """
        values = np.asarray(vatt, dtype=np.float64).reshape(-1)
        timestamps = np.asarray(cur, dtype=np.float64).reshape(-1)
        if values.size == 0 or timestamps.size != values.size:
            raise ValueError("R1 post-hoc requires matching nonempty key/timestamps.")
        values = (values - values.min()) / (values.max() - values.min() + 1e-12)
        peak = int(np.argmax(values))
        path = [peak]
        mean_s = np.mean(values)
        threshold = mean_s + thd_weight * mean_s
        stamp_s = timestamps[peak]

        i = 1
        while peak - i >= 0:  # ONLY change versus Metric.find_seg_ada
            stamp = timestamps[peak - i]
            elapse = abs(stamp_s - stamp)
            score = values[peak - i] / (1 + elapse / dis)
            if score >= threshold:
                path.append(peak - i)
            elif elapse > dis:
                break
            i += 1

        i = 1
        while peak + i < values.size:
            stamp = timestamps[peak + i]
            elapse = abs(stamp_s - stamp)
            score = values[peak + i] / (1 + elapse / dis)
            if score >= threshold:
                path.append(peak + i)
            elif elapse > dis:
                break
            i += 1
        return [float(timestamps[min(path)]), float(timestamps[max(path)])]

    @staticmethod
    def _r1_key_video_id(question_id):
        """NExT-GQA key: video_id_question_id, video_id may not contain '_'."""
        key = str(question_id)
        if "_" not in key:
            raise ValueError(f"Unexpected NExT-GQA question key: {key!r}")
        return key.split("_", 1)[0]

    @staticmethod
    def _r1_span_diagnostics(predictions, prefix):
        if not predictions:
            return {}
        spans = np.asarray(list(predictions.values()), dtype=np.float64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError("R1 span diagnostics expects predictions of shape [N,2].")
        length = spans[:, 1] - spans[:, 0]
        if np.any(length < -1e-6):
            raise RuntimeError("Temporal decoder emitted reversed interval endpoints.")
        return {
            f"{prefix}ZeroDurationPct": float(np.mean(np.abs(length) <= 1e-9) * 100.0),
            f"{prefix}MeanLengthSeconds": float(np.mean(length)),
        }

    def _train_epoch(self, epoch):
        self._h1_check_optimizer_contract()
        self._v37_check_optimizer_contract()
        self.monitor.log_info(
            f"\n{'#' * 20} Epoch "
            f"{epoch}/{self.cfgs['optim']['epochs']} "
            f"... {'#' * 20}\n"
        )
        print("Training Stage")
        self.model.train()

        self.monitor.reset_kv(
            f"{self.cfgs['stat']['monitor']['metric']}"
        )
        self.monitor.reset_kv(
            "total_loss"
        )

        for key in (
            self.cfgs["optim"]["loss"].keys()
        ):
            if key in {
                "ce",
                "align",
            }:
                continue
            self.monitor.reset_kv(
                f"{key}_loss"
            )

        if self._qbound_enabled():
            for key in self._qbound_monitor_keys():
                if not self.v35_pair_settings["enabled"] and key.startswith("qbound_pair_"):
                    continue
                if self.h1_temporal_mode != "eq_ptc" and "eqptc" in key:
                    continue
                if self.v37_event_mode != "event_relative" and (
                    "qbound_event_" in key
                ):
                    continue
                if not self.r2_predictor_enabled and (
                    "predictive" in key or "edge_surprise" in key
                    or "edge_affinity_raw" in key or "support_predictor_grad" in key
                ):
                    continue
                self.monitor.reset_kv(key)

        v35_valid_pairs_this_epoch = 0
        for index, batch in enumerate(
            self.train_dataloader
        ):
            start_time = time.time()

            (
                loss,
                monitor_metric,
                qbound_stats,
            ) = self.get_loss(
                batch,
                qbound_mix=(
                    1.0
                    if self._qbound_enabled()
                    else 0.0
                ),
            )

            self.monitor.logkv_mean(
                "acc",
                monitor_metric,
            )
            if self.v35_pair_settings["enabled"]:
                v35_valid_pairs_this_epoch += int(qbound_stats.get("qbound_pair_valid_count", 0))
            for name, value in (
                qbound_stats.items()
            ):
                self.monitor.logkv_mean(
                    name,
                    value,
                )

            self.optimizer.zero_grad()
            loss[
                "total_loss"
            ].backward()

            if self._qbound_enabled():
                if index == 0 and self.h1_temporal_mode == "eq_ptc":
                    core = self._h1_core()
                    for branch, prefixes in (
                        ("C-pre", ("eqptc_early_question_attn", "eqptc_early_delta")),
                        ("Bq", ("eqptc_temporal_attn", "eqptc_temporal_delta")),
                    ):
                        grads = [p.grad for name,p in core.named_parameters()
                                 if name.startswith(prefixes)]
                        if not grads or not any(g is not None and
                                                torch.isfinite(g).all() and
                                                bool((g.detach().abs()>0).any())
                                                for g in grads):
                            raise RuntimeError(
                                f"H1 EQ-PTC {branch} has no finite nonzero gradient "
                                "at training step 1. Inspect scales and loss path."
                            )
                if index == 0 and self.v37_event_mode == "event_relative":
                    core = self._h1_core()
                    comparator_grads = [p.grad for name, p in core.named_parameters()
                                        if name.startswith("event_comparator.")]
                    if not comparator_grads or not any(
                        g is not None and torch.isfinite(g).all()
                        and bool((g.detach().abs() > 0).any())
                        for g in comparator_grads
                    ):
                        raise RuntimeError(
                            "V3.7 Event Comparator has no finite nonzero gradient "
                            "from the training objective at step 1."
                        )
                self.monitor.logkv_mean(
                    "qbound_grad_norm",
                    self._qbound_grad_norm(),
                )
                for (
                    grad_name,
                    grad_value,
                ) in (
                    self._qbound_component_grad_norms()
                    .items()
                ):
                    self.monitor.logkv_mean(
                        grad_name,
                        grad_value,
                    )

            clip_value = (
                self.cfgs["optim"].get(
                    "clip",
                    0,
                )
            )
            if clip_value:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_value,
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(
                index + 1
            ) / len(
                self.train_dataloader
            )
            end_time = time.time()

            (
                _,
                remaining_batch_time,
                remaining_epoch_time,
            ) = get_remain_time(
                start_time,
                end_time,
                index,
                len(self.train_dataloader),
                epoch,
                self.cfgs["optim"]["epochs"],
            )

            print_info = (
                f"Progress: {progress:3.2%} "
                f"(rbt-ret: "
                f"{remaining_batch_time}-"
                f"{remaining_epoch_time}) | "
                f"Train "
                f"{self.cfgs['stat']['monitor']['metric']}: "
                f"{monitor_metric:3.2%}"
            )

            for (
                loss_name,
                loss_value,
            ) in loss.items():
                running_loss = (
                    loss_value
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                self.monitor.logkv_mean(
                    loss_name,
                    running_loss,
                )
                print_info += (
                    f" {loss_name}: "
                    f"{running_loss:.4f}"
                )

            print(
                f"\r{print_info}",
                end="",
                flush=True,
            )

        if self.v35_pair_settings["enabled"] and v35_valid_pairs_this_epoch <= 0:
            raise RuntimeError(
                "V3.5 enabled but no valid same-video question pairs were used "
                "during this epoch. Inspect dataset pairing/filter before training."
            )

        table = [[], []]
        for (
            name,
            value,
        ) in self.monitor.name2val.items():
            table[0].append(
                f"mean {name}"
            )
            table[1].append(
                f"{value:.4f}"
            )

        self.monitor.log_info(
            "\n" + tabulate(
                table,
                headers="firstrow",
                tablefmt="grid",
            ),
            end="\n",
        )

        print("Eval Stage: validation (canonical R2-C/V3.5 model selection)")
        val_score, val_result = self.eval(self.val_dataloader)
        val_result["r2_graph_mode"] = self.r2_graph_mode
        val_result["r2_predictor_enabled"] = self.r2_predictor_enabled
        val_result["r2_variant"] = ("A" if self.r2_graph_mode == "learned"
                                    else "B" if self.r2_predictor_enabled else "C")
        val_result["experiment_variant"] = (
            f"H1-{self.h1_temporal_mode}-QDTG" if self.v35_pair_settings["enabled"]
            else f"H1-{self.h1_temporal_mode}-R2C"
        )
        if self._qbound_enabled():
            val_result["experiment_variant"] += (
                f"-pool_{self.v37_pooling_mode}-event_{self.v37_event_mode}"
            )
        val_result["v37_language_pooling_mode"] = self.v37_pooling_mode
        val_result["v37_event_grounding_mode"] = self.v37_event_mode
        val_result["epoch"] = int(epoch)
        val_result["split"] = "val"

        if self.save_val_predictions:
            dump_dir = os.path.join(self.save_dir, "validation_dumps")
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"val_epoch_{epoch:02d}.pth")
            # Compatible with previously saved val_epoch_15.pth format:
            # {epoch, score, result}. Each per-question prediction is in result.
            torch.save(
                {
                    "epoch": int(epoch),
                    "score": val_score,
                    "result": val_result,
                    "r2_graph_mode": self.r2_graph_mode,
                    "r2_predictor_enabled": self.r2_predictor_enabled,
                    "r2_variant": val_result["r2_variant"],
                    "experiment_variant": val_result["experiment_variant"],
                },
                dump_path,
            )
            print(f"[R2 {self.r2_graph_mode}] Validation dump: {dump_path}")

        # Evaluate TEST every epoch for observation/WandB only. The checkpoint
        # selector in BasePipeline uses ONLY the validation score.
        if not self.eval_test_each_epoch:
            raise RuntimeError("V3.7 must not disable test evaluation each epoch.")
        test_score = {}
        if self.eval_test_each_epoch:
            print("Eval Stage: test (diagnostic only; NEVER for selection)")
            if self.cfgs["dataset"]["name"] == "star":
                test_score = val_score
            else:
                test_score, _ = self.eval(self.test_dataloader)

        score = {
            **{
                f"val_{key}": value
                for key, value
                in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value
                in test_score.items()
            },
        }
        self.print_score(score)

        wandb_data = {
            "epoch": epoch,
            "experiment/r2_is_fixed_graph": int(self.r2_graph_mode == "fixed")
            if self._qbound_enabled() else 0,
            "experiment/r2_predictor_enabled": int(self.r2_predictor_enabled)
            if self._qbound_enabled() else 0,
            "experiment/v35_pair_loss_enabled": int(self.v35_pair_settings["enabled"]),
            "experiment/v35_pair_loss_weight": float(self.v35_pair_settings["weight"]),
            "experiment/h1_eqptc_enabled": int(self.h1_temporal_mode == "eq_ptc"),
            "experiment/v37_event_relative_enabled": int(self.v37_event_mode == "event_relative"),
            "experiment/v37_masked_pooling_enabled": int(self.v37_pooling_mode == "masked"),
            "experiment/eval_test_each_epoch": 1,
        }
        for (
            name,
            value,
        ) in self.monitor.name2val.items():
            wandb_data[
                f"train/{name}"
            ] = value

        for (
            split_name,
            split_dict,
        ) in (
            ("val", val_score),
            ("test", test_score),
        ):
            for (
                key,
                value,
            ) in split_dict.items():
                if isinstance(
                    value,
                    dict,
                ):
                    for (
                        sub_key,
                        sub_value,
                    ) in value.items():
                        wandb_data[
                            f"{split_name}/"
                            f"{key}_{sub_key}"
                        ] = sub_value
                else:
                    wandb_data[
                        f"{split_name}/{key}"
                    ] = value

        wandb.log(
            wandb_data
        )
        # BasePipeline._record_best stores this as result.pth; it MUST be val.
        return score, val_result

    def print_score(self, score):
        def _print_score(score_dict):
            self.monitor.log_info("\n")
            for key in score_dict.keys():
                self.monitor.log_info(f"{key}  ")
            self.monitor.log_info("\n")
            for key in score_dict.keys():
                self.monitor.log_info(f"{score_dict[key]:.1f} \t ")

        for split in ("val", "test"):
            acc_key = f"{split}_acc"
            if acc_key in score:
                self.monitor.log_info(
                    f"\n{split.title()} Set\t "
                    f"Q-Bound R2-C/V3.5 acc: {score[acc_key]:.2%}"
                )

            prehoc_key = f"{split}_pre-hoc"
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} V2.8 posterior-barycentric =========="
                )
                _print_score(score[prehoc_key])

            posthoc_key = f"{split}_post-hoc"
            if posthoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} CRA post-hoc =========="
                )
                _print_score(score[posthoc_key])

            for diagnostic_name in (
                "pre-hoc-timestamp", "post-hoc-leftfix", "seed-post-hoc"
            ):
                diagnostic_key = f"{split}_{diagnostic_name}"
                if diagnostic_key in score:
                    self.monitor.log_info(
                        f"\n========== {split} {diagnostic_name} [diagnostic] =========="
                    )
                    _print_score(score[diagnostic_key])

            merge_key = f"{split}_merge"
            if merge_key in score:
                self.monitor.log_info(
                    f"\n========== {split} merge diagnostic =========="
                )
                _print_score(score[merge_key])

        qbound_keys = [
            key for key in score
            if key.startswith(("val_qbound_", "test_qbound_"))
        ]
        if qbound_keys:
            self.monitor.log_info(
                "\n================ Q-Bound V2.8 diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

        temporal_prefixes = (
            "val_QBoundSeed",
            "test_QBoundSeed",
            "val_QBoundSupport",
            "test_QBoundSupport",
            "val_CRA-",
            "test_CRA-",
        )
        temporal_keys = [
            key for key in score if key.startswith(temporal_prefixes)
        ]
        if temporal_keys:
            self.monitor.log_info(
                "\n============ Temporal support faithfulness ============"
            )
            for key in sorted(temporal_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

        print("\n")

    def draw_grounding(
        self,
        gt,
        posthoc,
        attn,
        merge,
        epoch,
        probs=None,
    ):
        image_path = os.path.join(
            self.save_dir,
            "record",
            "grounding",
        )
        os.makedirs(
            image_path,
            exist_ok=True,
        )

        video_question_ids = list(
            posthoc.keys()
        )
        sample_count = min(
            32,
            len(video_question_ids),
        )
        shuffle(video_question_ids)

        for index in range(
            sample_count
        ):
            video_question_id = (
                video_question_ids[index]
            )
            (
                video_id,
                question_id,
            ) = video_question_id.split(
                "_"
            )

            duration = gt[
                video_id
            ]["duration"]
            ground_truth = gt[
                video_id
            ]["location"][question_id]

            posthoc_ground = posthoc[
                video_question_id
            ]
            attention_ground = np.round(
                np.asarray(
                    attn[
                        video_question_id
                    ]
                )
                * duration,
                1,
            )
            merged_ground = merge[
                video_question_id
            ]

            figure, axis = plt.subplots()
            axis.plot(
                attention_ground,
                [3, 3],
                linewidth=6,
            )
            axis.plot(
                posthoc_ground,
                [2, 2],
                linewidth=6,
            )
            axis.plot(
                merged_ground,
                [1, 1],
                linewidth=6,
            )
            axis.plot(
                ground_truth[0],
                [0, 0],
                linewidth=6,
            )

            if probs is not None:
                probability = np.asarray(
                    probs[
                        video_question_id
                    ]
                ) * 3
                axis.plot(
                    np.linspace(
                        0,
                        duration,
                        len(probability),
                    ),
                    probability,
                    linewidth=3,
                    label="Temporal relevance",
                )

            plt.tight_layout()
            figure.savefig(
                os.path.join(
                    image_path,
                    f"{epoch}_"
                    f"{video_id}_"
                    f"{question_id}.png",
                ),
                dpi=300,
            )
            plt.close(figure)

    def eval(
        self,
        data_loader,
        epoch=None,
        save=False,
    ):
        del epoch, save

        self.model.eval()
        device = self._model_device()
        count = 0
        score = {"acc": 0.0}

        results = {}
        ground_res = {}
        ground_pre = {}
        keyframe = {}
        ori_keyframe = {}
        qbound_seed_keyframe = {}
        qbound_legacy_seed_keyframe = {}
        qbound_event_frame_keyframe = {}
        qbound_event_top_spans = {}
        qbound_event_valid_video_mask = {}
        qbound_support_keyframe = {}
        cra_keyframe = {}
        selected_frame_bounds = {}
        learned_edge_by_qid = {}
        used_edge_by_qid = {}
        edge_surprise_by_qid = {}

        qbound_stat_sums = collections.defaultdict(float)
        qbound_stat_count = 0
        saw_time_param = False

        with torch.no_grad():
            if not self.cfgs["dataset"]["mc"]:
                model_core = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                if not hasattr(model_core, "_compute_answer_embedding"):
                    raise AttributeError(
                        "Non-multiple-choice evaluation requires "
                        "_compute_answer_embedding on the model."
                    )
                model_core._compute_answer_embedding(self.a2v)

            for index, batch in enumerate(data_loader):
                answer_id = batch["answer_id"]
                answer = batch["answer"].to(device, non_blocking=True)
                video_frames = batch["video_frames"].to(
                    device, non_blocking=True
                )
                question = batch["question"].to(
                    device, non_blocking=True
                )
                question_id = batch["question_id"]
                video_len = batch["video_len"]
                video_mask = get_mask(
                    video_len, video_frames.size(1)
                ).to(device, non_blocking=True)

                batch_size = answer_id.size(0)
                count += batch_size
                labels_cpu = answer_id.reshape(-1).cpu()

                gqa_output = self.model(
                    video_frames,
                    video_mask,
                    answer=answer,
                    answer_id=answer_id,
                    mode="VQA",
                    question=question,
                    qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
                    answer_only=None,
                )
                fusion_proj, answer_proj, model_outputs = gqa_output
                main_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=fusion_proj,
                    detach_answer=False,
                )
                predicted = main_logits.argmax(dim=-1).cpu()
                score["acc"] += (
                    predicted == labels_cpu
                ).sum().item()

                time_param = model_outputs.get("time_param")
                qbound_aux = self._get_qbound_aux(gqa_output)
                if isinstance(time_param, dict):
                    saw_time_param = True

                if self._qbound_enabled():
                    qbound_aux = self._validate_v280_contract(gqa_output)
                    batch_stats = self._collect_qbound_stats(gqa_output)
                    for name, value in batch_stats.items():
                        qbound_stat_sums[name] += float(value) * batch_size
                    qbound_stat_count += batch_size

                video_attention = model_outputs["fatt"].detach().cpu().numpy()
                if video_attention.ndim == 3 and video_attention.shape[-1] == 1:
                    video_attention = video_attention.squeeze(-1)
                if video_attention.ndim == 1:
                    video_attention = video_attention[None, :]

                batch_ground_pre = None
                batch_keyframe = None
                batch_ori_keyframe = None
                batch_seed_key = None
                batch_legacy_seed_key = None
                batch_event_frame_key = None
                batch_event_top_starts = None
                batch_event_top_ends = None
                batch_event_video_mask = None
                batch_support_key = None
                batch_cra_key = None
                batch_frame_bounds = None
                batch_learned_edge = None
                batch_used_edge = None
                batch_edge_surprise = None

                if isinstance(time_param, dict):
                    start_time = time_param["start"].detach().cpu().numpy()
                    end_time = time_param["end"].detach().cpu().numpy()
                    batch_ground_pre = np.column_stack((start_time, end_time))
                    batch_keyframe = time_param["key"].detach().cpu().numpy()
                    batch_ori_keyframe = (
                        time_param["ori_key"].detach().cpu().numpy()
                    )

                if qbound_aux is not None:
                    seed_key = qbound_aux.get("semantic_seed_prob")
                    support_key = qbound_aux.get("support_temporal_prob")
                    cra_key = qbound_aux.get("cra_key")
                    mask = qbound_aux.get("video_mask")
                    # Preserve both learned edges and edges actually passed to
                    # propagation. This distinguishes graph learning from smoothing.
                    learned_edge = qbound_aux.get("support_edge_affinity_raw")
                    used_edge = qbound_aux.get("support_edge_affinity")
                    surprise = qbound_aux.get("support_edge_surprise")
                    if self.r2_predictor_enabled and torch.is_tensor(learned_edge):
                        batch_learned_edge = learned_edge.detach().cpu().numpy()
                    if torch.is_tensor(used_edge):
                        batch_used_edge = used_edge.detach().cpu().numpy()
                    if self.r2_predictor_enabled and torch.is_tensor(surprise):
                        batch_edge_surprise = surprise.detach().cpu().numpy()
                    start_idx = qbound_aux.get("selected_span_start_frame")
                    end_idx = qbound_aux.get("selected_span_end_frame")
                    if torch.is_tensor(start_idx) and torch.is_tensor(end_idx):
                        batch_frame_bounds = np.column_stack((
                            start_idx.detach().cpu().numpy(),
                            end_idx.detach().cpu().numpy(),
                        ))
                    if self.v37_event_mode == "event_relative":
                        legacy_seed = qbound_aux["legacy_semantic_seed_prob"]
                        event_frames = qbound_aux["event_frame_prob"]
                        top_starts = qbound_aux["event_candidate_top_start_frame"]
                        top_ends = qbound_aux["event_candidate_top_end_frame"]
                        batch_legacy_seed_key = self._normalize_distribution(
                            legacy_seed.detach(), mask=mask
                        ).cpu().numpy()
                        batch_event_frame_key = self._normalize_distribution(
                            event_frames.detach(), mask=mask
                        ).cpu().numpy()
                        batch_event_top_starts = top_starts.detach().cpu().numpy()
                        batch_event_top_ends = top_ends.detach().cpu().numpy()
                        batch_event_video_mask = mask.detach().bool().cpu().numpy()
                    if torch.is_tensor(seed_key):
                        batch_seed_key = self._normalize_distribution(
                            seed_key.detach(), mask=mask
                        ).cpu().numpy()
                    if torch.is_tensor(support_key):
                        batch_support_key = self._normalize_distribution(
                            support_key.detach(), mask=mask
                        ).cpu().numpy()
                    if torch.is_tensor(cra_key):
                        batch_cra_key = self._normalize_distribution(
                            cra_key.detach(), mask=mask
                        ).cpu().numpy()

                # The original CRA post-hoc decoder must see the same Support
                # Field used for ground/QA. Any future divergence invalidates R1/R2.
                if self._qbound_enabled() and batch_support_key is not None:
                    if not np.allclose(
                        video_attention, batch_support_key, atol=1e-5, rtol=1e-5
                    ):
                        raise RuntimeError(
                            "R2 protocol broken: model_outputs['fatt'] differs "
                            "from the Q-Bound Support Field used by QA."
                        )
                predicted_np = predicted.numpy()
                labels_np = labels_cpu.numpy()
                for batch_index, qid in enumerate(question_id):
                    results[qid] = {
                        "prediction": int(predicted_np[batch_index]),
                        "answer": int(labels_np[batch_index]),
                    }
                    ground_res[qid] = video_attention[batch_index].tolist()
                    if batch_ground_pre is not None:
                        ground_pre[qid] = batch_ground_pre[batch_index].tolist()
                    if batch_keyframe is not None:
                        keyframe[qid] = batch_keyframe[batch_index].tolist()
                    if batch_ori_keyframe is not None:
                        ori_keyframe[qid] = batch_ori_keyframe[batch_index].tolist()
                    if batch_seed_key is not None:
                        qbound_seed_keyframe[qid] = (
                            batch_seed_key[batch_index].tolist()
                        )
                    if batch_legacy_seed_key is not None:
                        qbound_legacy_seed_keyframe[qid] = (
                            batch_legacy_seed_key[batch_index].tolist()
                        )
                        qbound_event_frame_keyframe[qid] = (
                            batch_event_frame_key[batch_index].tolist()
                        )
                        qbound_event_top_spans[qid] = [
                            int(batch_event_top_starts[batch_index]),
                            int(batch_event_top_ends[batch_index]),
                        ]
                        qbound_event_valid_video_mask[qid] = (
                            batch_event_video_mask[batch_index].tolist()
                        )
                    if batch_support_key is not None:
                        qbound_support_keyframe[qid] = (
                            batch_support_key[batch_index].tolist()
                        )
                    if batch_cra_key is not None:
                        cra_keyframe[qid] = batch_cra_key[batch_index].tolist()
                    if batch_frame_bounds is not None:
                        selected_frame_bounds[qid] = [
                            int(batch_frame_bounds[batch_index][0]),
                            int(batch_frame_bounds[batch_index][1]),
                        ]
                    if batch_learned_edge is not None:
                        learned_edge_by_qid[qid] = batch_learned_edge[batch_index].tolist()
                    if batch_used_edge is not None:
                        used_edge_by_qid[qid] = batch_used_edge[batch_index].tolist()
                    if batch_edge_surprise is not None:
                        edge_surprise_by_qid[qid] = batch_edge_surprise[batch_index].tolist()

                progress = float(index + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end="", flush=True)

        if count == 0:
            raise RuntimeError("Evaluation data loader is empty.")
        score["acc"] /= count
        if qbound_stat_count > 0:
            for name, total in qbound_stat_sums.items():
                score[name] = total / qbound_stat_count

        split = data_loader.dataset.split
        csv_path = self.cfgs["dataset"]["csv_path"]
        ground_truth = load_file(
            os.path.join(csv_path, f"gsub_{split}.json")
        )
        segments = load_file(
            os.path.join(csv_path, f"frame2time_{split}.json")
        )
        questions = load_file(os.path.join(csv_path, f"{split}.csv"))
        # H1 analysis is STRICTLY post-inference, validation/test only.
        # Never provide temporal GT to QBoundCore or get_loss().
        h1_metrics = (
            self._h1_gt_evidence_metrics(
                qbound_seed_keyframe, qbound_support_keyframe,
                ground_truth, segments,
                question_csv_path=os.path.join(csv_path, f"{split}.csv")
            ) if self._qbound_enabled() else {}
        )
        score.update(h1_metrics)

        post_hoc_ground = self.metric.generate_ground(
            ground_res, segments, questions
        )
        score["post-hoc"] = self.metric(
            post_hoc_ground, ground_truth, results, gs=False
        )
        if self._qbound_enabled():
            # R1 diagnostics ONLY. The canonical post-hoc decoder above is
            # unchanged and remains the sole checkpoint-selection reference.
            post_hoc_leftfix = {}
            for qid in post_hoc_ground:
                vid = self._r1_key_video_id(qid)
                post_hoc_leftfix[qid] = self._r1_posthoc_frame0_fixed(
                    ground_res[qid], segments[vid]
                )
            score["post-hoc-leftfix"] = self.metric(
                post_hoc_leftfix, ground_truth, results, gs=False
            )
            score.update(self._r1_span_diagnostics(
                post_hoc_ground, "QBoundPosthoc"
            ))

            seed_post_hoc = self.metric.generate_ground(
                qbound_seed_keyframe, segments, questions
            )
            score["seed-post-hoc"] = self.metric(
                seed_post_hoc, ground_truth, results, gs=False
            )
            score.update(self._r1_span_diagnostics(
                seed_post_hoc, "QBoundSeedPosthoc"
            ))
        else:
            post_hoc_leftfix = {}
            seed_post_hoc = {}

        if qbound_support_keyframe:
            if not hasattr(self.metric, "temporal_key_metrics"):
                raise AttributeError(
                    "Q-Bound V2.8 requires Metric.temporal_key_metrics()."
                )
            seed_metrics = self.metric.temporal_key_metrics(
                qbound_seed_keyframe,
                ground_truth,
                segments,
                prefix="QBoundSeed",
            )
            support_metrics = self.metric.temporal_key_metrics(
                qbound_support_keyframe,
                ground_truth,
                segments,
                prefix="QBoundSupport",
            )
            score.update(seed_metrics)
            score.update(support_metrics)

            # Directly expose whether structural propagation improves each
            # temporal metric over the semantic seed.
            for support_name, support_value in support_metrics.items():
                suffix = support_name[len("QBoundSupport"):]
                seed_name = f"QBoundSeed{suffix}"
                seed_value = seed_metrics.get(seed_name)
                if isinstance(support_value, (int, float)) and isinstance(
                    seed_value, (int, float)
                ):
                    score[f"QBoundSupportGain{suffix}"] = (
                        float(support_value) - float(seed_value)
                    )

        if self._qbound_enabled() and self.v37_event_mode == "event_relative":
            # This is post-inference validation/test analysis ONLY: never use
            # temporal GT in the model, training loss or checkpoint ranking.
            if len(qbound_event_frame_keyframe) != len(qbound_seed_keyframe):
                raise RuntimeError("V3.7 evaluation is missing event-frame predictions.")
            score.update(self.metric.temporal_key_metrics(
                qbound_legacy_seed_keyframe,
                ground_truth, segments, prefix="QBoundLegacySeed",
            ))
            score.update(self.metric.temporal_key_metrics(
                qbound_event_frame_keyframe,
                ground_truth, segments, prefix="QBoundEventFrame",
            ))
            # Distinguish occurrence selection from mere event-score movement.
            score.update(self._v37_candidate_gt_diagnostics(
                qbound_event_top_spans, qbound_event_valid_video_mask,
                ground_truth, segments,
            ))

        if cra_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    cra_keyframe,
                    ground_truth,
                    segments,
                    prefix="CRA",
                )
            )

        result = {
            "ph": post_hoc_ground,
            "result": results,
            "gt_ground": ground_truth,
            "ph_attn": ground_res,
            "qbound_seed_key": qbound_seed_keyframe,
            "qbound_legacy_seed_key": qbound_legacy_seed_keyframe,
            "qbound_event_frame_key": qbound_event_frame_keyframe,
            "qbound_event_top_spans": qbound_event_top_spans,
            "qbound_event_grounding_mode": self.v37_event_mode,
            "qbound_language_pooling_mode": self.v37_pooling_mode,
            "qbound_support_key": qbound_support_keyframe,
            # Compatibility: qbound_key now means the continuous support.
            "qbound_key": qbound_support_keyframe,
            "cra_key": cra_keyframe,
            "ph_r1_leftfix": post_hoc_leftfix,
            "ph_seed": seed_post_hoc,
            "qbound_edge_affinity_learned": learned_edge_by_qid,
            "qbound_edge_affinity_used": used_edge_by_qid,
            "qbound_edge_surprise": edge_surprise_by_qid,
            "qbound_graph_mode": self.r2_graph_mode,
            "h1_evidence_metrics": h1_metrics,
            "h1_temporal_context_mode": self.h1_temporal_mode,
        }

        if saw_time_param and ground_pre:
            score["pre-hoc"] = self.metric(
                ground_pre, ground_truth, results, gs=True
            )
            result.update(
                {
                    "tg": ground_pre,
                    "qbound_tg": ground_pre,
                    "key": keyframe,
                    "ori_key": ori_keyframe,
                }
            )

            merge_ground = self.metric.combine(
                ground_pre, post_hoc_ground, ground_truth
            )
            score["merge"] = self.metric(
                merge_ground, ground_truth, results, gs=False
            )
            result["merge"] = merge_ground

            # R1 timestamp control: use SAME rounded barycenter frame indices
            # that generated tg, mapped to actual sampled-frame timestamps.
            # Do not change the canonical normalized-time pre-hoc metric.
            if self._qbound_enabled():
                ground_pre_timestamp = {}
                for qid, bounds in selected_frame_bounds.items():
                    vid = self._r1_key_video_id(qid)
                    times = np.asarray(segments[vid], dtype=np.float64).reshape(-1)
                    support_size = len(qbound_support_keyframe[qid])
                    if times.size != support_size:
                        raise ValueError(
                            f"R1 timestamps vs Support Field mismatch at {qid}: "
                            f"{times.size} timestamps, {support_size} frames."
                        )
                    start_idx, end_idx = bounds
                    if not (0 <= start_idx <= end_idx < times.size):
                        raise RuntimeError(f"Invalid barycenter indices for {qid}.")
                    ground_pre_timestamp[qid] = [
                        float(times[start_idx]), float(times[end_idx])
                    ]
                if len(ground_pre_timestamp) != len(ground_pre):
                    raise RuntimeError(
                        "R1 timestamp diagnostic lacks some pre-hoc predictions."
                    )
                score["pre-hoc-timestamp"] = self.metric(
                    ground_pre_timestamp, ground_truth, results, gs=False
                )
                result["tg_timestamp"] = ground_pre_timestamp

        return score, result

    def infer(self, epoch=None):
        parameter_count = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )
        print(
            f"Model Params: "
            f"{parameter_count}"
        )
        print("\nEval Stage")

        val_score, _ = self.eval(
            self.val_dataloader,
            epoch,
            self.cfgs["dataset"]["name"]
            == "star",
        )

        if (
            self.cfgs["dataset"]["name"]
            == "star"
        ):
            test_score = val_score
        else:
            test_score, _ = self.eval(
                self.test_dataloader,
                epoch,
                True,
            )

        score = {
            **{
                f"val_{key}": value
                for key, value
                in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value
                in test_score.items()
            },
        }

        self.print_score(score)
        return score
