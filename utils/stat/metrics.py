from typing import Any, Dict, Optional, Sequence

import numpy as np


class Metric(object):
    """
    Original CRA / NExT-GQA grounding metrics + Q-Bound Ver1.4 diagnostics.

    Benchmark metrics are intentionally preserved:
        Acc&GQA
        Acc&VQA
        mIoP
        TIoP@0.3 / TIoP@0.5
        mIoU
        TIoU@0.3 / TIoU@0.5

    Q-Bound additionally exposes temporal-key diagnostics through
    ``temporal_key_metrics()``:
        Key-GT-Mass
        Key-Peak-in-GT
        Key-Center-Error
        Key-Peak-Center-Error
        Key-Entropy-Ratio
        Key-Peak-Ratio

    V1.4 decomposition/reasoning health statistics (partition energy, event
    count/mass, split gain, Q-OEI gates, multi-event evidence, etc.) are logged
    by CRAPipeline and intentionally do NOT alter these benchmark definitions.
    """

    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs

    def __call__(
        self,
        pred_ground,
        target_ground,
        pred_qa=None,
        subset=None,
        gs=False,
    ):
        # Keep official CRA/NExT-GQA metric semantics unchanged.
        mIoU, mIoP = 0.0, 0.0
        cnt, cqt, cqtplus = 0, 0, 0
        crt3, crt5 = 0, 0
        crtp3, crtp5 = 0, 0
        cqt_5, cqt_3 = 0, 0
        ufcqt_5, ufcqt_3 = 0, 0

        for vid, anno in target_ground.items():
            for qid, locs in anno["location"].items():
                kid = f"{vid}_{qid}"

                if kid not in pred_ground:
                    continue

                if subset is not None and kid not in subset:
                    continue

                max_tIoU, max_tIoP = 0.0, 0.0

                for loc in locs:
                    span = pred_ground[kid]

                    if gs:
                        span = np.round(
                            np.asarray(span) * anno["duration"],
                            1,
                        )

                    tIoU, tIoP = self.get_tIoU(loc, span)
                    max_tIoU = max(max_tIoU, tIoU)
                    max_tIoP = max(max_tIoP, tIoP)

                if max_tIoP >= 0.3:
                    crtp3 += 1

                    if max_tIoP >= 0.5:
                        crtp5 += 1

                        if pred_qa:
                            if (
                                pred_qa[kid]["answer"]
                                == pred_qa[kid]["prediction"]
                            ):
                                cqt += 1

                if max_tIoU >= 0.3:
                    crt3 += 1

                    if max_tIoU >= 0.5:
                        crt5 += 1

                if max_tIoP < 0.5 and pred_qa:
                    if (
                        pred_qa[kid]["answer"]
                        == pred_qa[kid]["prediction"]
                    ):
                        ufcqt_5 += 1
                    else:
                        cqt_5 += 1

                    if max_tIoP < 0.3:
                        if (
                            pred_qa[kid]["answer"]
                            == pred_qa[kid]["prediction"]
                        ):
                            ufcqt_3 += 1
                        else:
                            cqt_3 += 1

                if pred_qa:
                    if (
                        pred_qa[kid]["answer"]
                        == pred_qa[kid]["prediction"]
                    ):
                        cqtplus += 1

                cnt += 1
                mIoU += max_tIoU
                mIoP += max_tIoP

        if cnt == 0:
            return {
                "Acc&GQA": 0.0,
                "Acc&VQA": 0.0,
                "mIoP": 0.0,
                "TIoP@0.3": 0.0,
                "TIoP@0.5": 0.0,
                "mIoU": 0.0,
                "TIoU@0.3": 0.0,
                "TIoU@0.5": 0.0,
            }

        score = {
            "Acc&GQA": cqt * 1.0 / cnt * 100.0,
            "Acc&VQA": cqtplus * 1.0 / cnt * 100.0,
            "mIoP": mIoP / cnt * 100.0,
            "TIoP@0.3": crtp3 * 1.0 / cnt * 100.0,
            "TIoP@0.5": crtp5 * 1.0 / cnt * 100.0,
            "mIoU": mIoU / cnt * 100.0,
            "TIoU@0.3": crt3 * 1.0 / cnt * 100.0,
            "TIoU@0.5": crt5 * 1.0 / cnt * 100.0,
        }

        return score

    @staticmethod
    def get_tIoU(loc, span):
        if span[0] == span[-1]:
            if loc[0] <= span[0] <= loc[1]:
                return 0.0, 1.0
            return 0.0, 0.0

        span_u = (
            min(loc[0], span[0]),
            max(loc[-1], span[-1]),
        )
        span_i = (
            max(loc[0], span[0]),
            min(loc[-1], span[-1]),
        )

        dis_i = max(
            float(span_i[1] - span_i[0]),
            0.0,
        )

        if span_u[1] > span_u[0]:
            IoU = dis_i / float(span_u[1] - span_u[0])
        else:
            IoU = 0.0

        if span[-1] > span[0]:
            IoP = dis_i / float(span[-1] - span[0])
        else:
            IoP = 0.0

        return IoU, IoP

    # ------------------------------------------------------------------
    # Q-Bound Ver1.4 temporal-key diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_key(key, eps=1e-12):
        key = np.asarray(key, dtype=np.float64).reshape(-1)

        if key.size == 0:
            return key

        key = np.clip(key, 0.0, None)
        mass = float(key.sum())

        if mass <= eps:
            return np.full(
                key.shape,
                1.0 / float(key.size),
                dtype=np.float64,
            )

        return key / mass

    @staticmethod
    def _nearest_segment_distance(time_value, locs):
        best = float("inf")

        for loc in locs:
            start = float(loc[0])
            end = float(loc[-1])

            if start <= time_value <= end:
                return 0.0

            if time_value < start:
                distance = start - time_value
            else:
                distance = time_value - end

            best = min(best, distance)

        return 0.0 if not np.isfinite(best) else best

    @staticmethod
    def _timestamp_inside_any_gt(time_value, locs):
        for loc in locs:
            if float(loc[0]) <= time_value <= float(loc[-1]):
                return True
        return False

    @staticmethod
    def _fit_timestamps_to_key(timestamps, key_length, duration):
        timestamps = np.asarray(
            timestamps,
            dtype=np.float64,
        ).reshape(-1)

        if key_length <= 0:
            return np.asarray([], dtype=np.float64)

        if timestamps.size == key_length:
            return timestamps

        if timestamps.size >= 2:
            return np.linspace(
                float(timestamps[0]),
                float(timestamps[-1]),
                key_length,
                dtype=np.float64,
            )

        return np.linspace(
            0.0,
            float(duration),
            key_length,
            dtype=np.float64,
        )

    def temporal_key_metrics(
        self,
        key_predictions: Dict[str, Sequence[float]],
        target_ground: Dict[str, Any],
        frame2time: Dict[str, Any],
        subset: Optional[Any] = None,
        prefix: str = "QBound",
    ) -> Dict[str, float]:
        """
        Diagnose whether a frame-level temporal probability distribution
        actually overlaps the annotated evidence.

        These metrics are diagnostics only; they do not alter official CRA
        benchmark scores.
        """
        gt_mass_sum = 0.0
        peak_in_gt_count = 0
        center_error_sum = 0.0
        peak_center_error_sum = 0.0
        entropy_ratio_sum = 0.0
        peak_ratio_sum = 0.0
        count = 0

        for vid, anno in target_ground.items():
            duration = float(anno.get("duration", 0.0))

            if duration <= 0.0 or vid not in frame2time:
                continue

            for qid, locs in anno["location"].items():
                kid = f"{vid}_{qid}"

                if kid not in key_predictions:
                    continue

                if subset is not None and kid not in subset:
                    continue

                key = self._normalize_key(
                    key_predictions[kid]
                )

                if key.size == 0:
                    continue

                timestamps = self._fit_timestamps_to_key(
                    frame2time[vid],
                    key.size,
                    duration,
                )

                inside_gt = np.asarray(
                    [
                        self._timestamp_inside_any_gt(
                            float(timestamp),
                            locs,
                        )
                        for timestamp in timestamps
                    ],
                    dtype=bool,
                )

                gt_mass = float(
                    key[inside_gt].sum()
                )

                peak_index = int(
                    np.argmax(key)
                )
                peak_time = float(
                    timestamps[peak_index]
                )

                peak_in_gt = (
                    self._timestamp_inside_any_gt(
                        peak_time,
                        locs,
                    )
                )

                expected_time = float(
                    np.sum(
                        key * timestamps
                    )
                )

                center_error = (
                    self._nearest_segment_distance(
                        expected_time,
                        locs,
                    )
                    / max(duration, 1e-12)
                )

                peak_center_error = (
                    self._nearest_segment_distance(
                        peak_time,
                        locs,
                    )
                    / max(duration, 1e-12)
                )

                entropy = -float(
                    np.sum(
                        key
                        * np.log(
                            np.clip(
                                key,
                                1e-12,
                                None,
                            )
                        )
                    )
                )

                uniform_entropy = np.log(
                    max(key.size, 2)
                )

                entropy_ratio = (
                    entropy
                    / max(
                        uniform_entropy,
                        1e-12,
                    )
                )

                peak_ratio = float(
                    key.max() * key.size
                )

                gt_mass_sum += gt_mass
                peak_in_gt_count += int(
                    peak_in_gt
                )
                center_error_sum += (
                    center_error
                )
                peak_center_error_sum += (
                    peak_center_error
                )
                entropy_ratio_sum += (
                    entropy_ratio
                )
                peak_ratio_sum += (
                    peak_ratio
                )
                count += 1

        if count == 0:
            return {
                f"{prefix}-GT-Mass": 0.0,
                f"{prefix}-Peak-in-GT": 0.0,
                f"{prefix}-Center-Error": 0.0,
                f"{prefix}-Peak-Center-Error": 0.0,
                f"{prefix}-Entropy-Ratio": 0.0,
                f"{prefix}-Peak-Ratio": 0.0,
            }

        return {
            f"{prefix}-GT-Mass": (
                gt_mass_sum / count * 100.0
            ),
            f"{prefix}-Peak-in-GT": (
                peak_in_gt_count / count * 100.0
            ),
            f"{prefix}-Center-Error": (
                center_error_sum / count * 100.0
            ),
            f"{prefix}-Peak-Center-Error": (
                peak_center_error_sum / count * 100.0
            ),
            f"{prefix}-Entropy-Ratio": (
                entropy_ratio_sum / count
            ),
            f"{prefix}-Peak-Ratio": (
                peak_ratio_sum / count
            ),
        }

    @staticmethod
    def combine(pred1, pred2, gt, way="itsc"):
        def _cb_seg(seg1, seg2, way="itsc"):
            if way == "uni":
                ts = [
                    seg1[0],
                    seg1[1],
                    seg2[0],
                    seg2[1],
                ]
                ts = sorted(ts)
                new_seg = [ts[0], ts[-1]]

            elif way == "itsc":
                start = (
                    seg1[0]
                    if seg1[0] > seg2[0]
                    else seg2[0]
                )
                end = (
                    seg1[1]
                    if seg1[1] < seg2[1]
                    else seg2[1]
                )

                if not (start <= end):
                    new_seg = seg2.tolist()
                else:
                    new_seg = [start, end]

            elif way == "itsc_refine":
                start = (
                    seg1[0]
                    if seg1[0] > seg2[0]
                    else seg2[0]
                )
                end = (
                    seg1[1]
                    if seg1[1] < seg2[1]
                    else seg2[1]
                )

                if not (start <= end):
                    new_seg = seg1.tolist()
                else:
                    new_seg = [start, end]

            elif way == "itsc_swap":
                start = (
                    seg1[0]
                    if seg1[0] > seg2[0]
                    else seg2[0]
                )
                end = (
                    seg1[1]
                    if seg1[1] < seg2[1]
                    else seg2[1]
                )

                if not (start <= end):
                    new_seg = [end, start]
                else:
                    new_seg = [start, end]

            elif way == "itsc_mean":
                start = (
                    seg1[0]
                    if seg1[0] > seg2[0]
                    else seg2[0]
                )
                end = (
                    seg1[1]
                    if seg1[1] < seg2[1]
                    else seg2[1]
                )

                if not (start <= end):
                    new_seg = [
                        (seg1[0] + seg2[0]) // 2,
                        (seg1[1] + seg2[1]) // 2,
                    ]
                else:
                    new_seg = [start, end]

            else:
                raise ValueError(
                    f"Unknown combine way: {way!r}."
                )

            return new_seg

        cb_ground = {}

        for vqid, seg in pred1.items():
            if len(vqid.split("_")) > 2:
                vid = vqid.split("_")[0]
                qid = "_".join(
                    vqid.split("_")[1:]
                )
            else:
                vid, qid = vqid.split("_")

            if (
                vid not in gt
                or qid not in gt[vid]["location"]
            ):
                continue

            duration = gt[vid]["duration"]
            seg = np.round(
                np.asarray(seg) * duration,
                1,
            )
            seg_att = np.asarray(
                pred2[vqid]
            )

            cb_ground[vqid] = _cb_seg(
                seg,
                seg_att,
                way=way,
            )

        return cb_ground

    def generate_ground(
        self,
        preds,
        segs,
        qas,
        thd_weight=0.3,
        dis=10,
    ):
        res_ground = {}

        for _, row in qas.iterrows():
            vid = str(row["video_id"])
            qid = str(row["qid"])
            vid_qid = "_".join(
                [vid, qid]
            )

            if vid_qid not in preds:
                continue

            atts = preds[vid_qid]
            cur = np.asarray(
                segs[vid]
            )

            res_ground[vid_qid] = (
                self.find_seg_ada(
                    atts,
                    cur,
                    thd_weight,
                    dis,
                )
            )

        return res_ground

    @staticmethod
    def find_seg_win(vatt):
        window_sizes = [3, 5, 7]
        preds = np.array(
            vatt.copy()
        ).reshape([1, -1])

        _, length = preds.shape

        max_probs = np.zeros(1)
        max_indices = np.zeros(
            [1, 2]
        )

        for window_size in window_sizes:
            for i in range(
                length - window_size + 1
            ):
                window_probs = (
                    preds[
                        :,
                        i : i + window_size,
                    ].sum(axis=1)
                )

                window_max = (
                    preds[
                        :,
                        i : i + window_size,
                    ].max(axis=1)
                )

                overall_max = (
                    preds.max(axis=1)
                )

                valid_mask = (
                    window_max
                    == overall_max.squeeze()
                )

                max_mask = (
                    (window_probs > max_probs)
                    & valid_mask
                )

                max_probs[max_mask] = (
                    window_probs[max_mask]
                )
                max_indices[
                    max_mask,
                    0,
                ] = i
                max_indices[
                    max_mask,
                    1,
                ] = (
                    i + window_size - 1
                )

        denominator = float(
            max(length - 1, 1)
        )

        start = (
            max_indices[:, 0]
            / denominator
        )[0]

        end = (
            max_indices[:, 1]
            / denominator
        )[0]

        return [start, end]

    @staticmethod
    def find_seg_ada(
        vatt,
        cur,
        thd_weight=0.3,
        dis=10,
    ):
        if not isinstance(vatt, list):
            vatt = [vatt]

        n = len(vatt)
        vatt = np.asarray(
            vatt,
            dtype=np.float64,
        )

        vatt = (
            vatt - np.min(vatt)
        ) / (
            np.max(vatt)
            - np.min(vatt)
            + 1e-12
        )

        max_id = int(
            np.argmax(vatt)
        )
        mean_s = np.mean(vatt)
        path = [max_id]
        stamp_s = cur[max_id]

        i = 1
        thd = (
            mean_s
            + thd_weight * mean_s
        )

        while max_id - i > 0:
            stamp = cur[
                max_id - i
            ]

            elapse = abs(
                stamp_s - stamp
            )

            score = (
                vatt[max_id - i]
                / (1 + elapse / dis)
            )

            if score >= thd:
                path.append(
                    max_id - i
                )
            elif elapse > dis:
                break

            i += 1

        i = 1

        while max_id + i < n:
            stamp = cur[
                max_id + i
            ]

            elapse = abs(
                stamp_s - stamp
            )

            score = (
                vatt[max_id + i]
                / (1 + elapse / dis)
            )

            if score >= thd:
                path.append(
                    max_id + i
                )
            elif elapse > dis:
                break

            i += 1

        sid = int(np.min(path))
        eid = int(np.max(path))

        return [
            cur[sid],
            cur[eid],
        ]
