"""Parameter-free Q-Bound V3.8-M2 interval decoding and locked replay references.

No ground truth, QA label, model parameters, or CRA answer selection lives here.
Intervals are half-open in steps; their seconds endpoints are [time[s], time[e-1]].
The 496-column contract is for the standard 32-valid-step cohort.
"""

import re
import numpy as np


T = 32
WIDTHS = (2, 4, 8, 16)
BANK_BOUNDS = np.asarray(
    [(s, s + w) for w in WIDTHS for s in range(0, T - w + 1, w // 2)],
    dtype=np.int64,
)
SPAN_BOUNDS = np.asarray(
    [(s, e) for s in range(T) for e in range(s + 2, T + 1)],
    dtype=np.int64,
)
N_BANK = len(BANK_BOUNDS)
N_SPANS = len(SPAN_BOUNDS)
_BS, _BE = BANK_BOUNDS.T
_S, _E = SPAN_BOUNDS.T
_T = np.arange(T)
_FOOT = ((_T >= _BS[:, None]) & (_T < _BE[:, None])) / (_BE - _BS)[:, None]
_PINV = np.linalg.pinv(_FOOT, rcond=1e-10)
_CENTER = (_S + _E - 1) / 62.0
_BEGIN = {"beginning", "begin", "start", "started", "first"}
_END = {"end", "ending", "finally", "last"}


def timestamp_status(timestamps):
    """Classify actual sampled-frame centers WITHOUT fabricating or sorting times.

    Exact adjacent ties are permitted; they can arise when the sampled indices
    are repeated. The 32-step feature order is never changed. All-equal values
    cannot define a positive-duration span, so they remain an explicit error.
    """
    if timestamps is None:
        return "missing_key", None
    try:
        t = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return "not_numeric", None
    if t.shape not in ((T,), (T, 1), (1, T)):
        return "wrong_shape", None
    t = t.reshape(T)
    if not np.isfinite(t).all():
        return "nonfinite", None
    if np.any(t < 0):
        return "negative", None
    dt = np.diff(t)
    if np.any(dt < 0):
        return "decreasing", None
    if not np.any(dt > 0):
        return "all_equal", None
    return ("repeated_centers" if np.any(dt == 0) else "strict"), t


def validate_timestamps(timestamps):
    """Validate exactly 32 real, ordered sampled-frame centers.

    Non-decreasing ties are accepted and handled by select() with an explicit
    positive-duration span mask; missing/invalid centers never receive guessed
    timestamps. Strictly increasing inputs follow the ORIGINAL MBR code path.
    """
    status, t = timestamp_status(timestamps)
    if t is None:
        raise ValueError(f"M2 MBR timestamp error: {status}")
    return t


def _utility(t):
    lo, hi = t[_S], t[_E - 1]
    length = hi - lo
    inter = np.maximum(
        0.0, np.minimum(hi[:, None], hi[None, :]) - np.maximum(lo[:, None], lo[None, :])
    )
    union = length[:, None] + length[None, :] - inter
    # Rows are proposed outputs; columns are hypothetical target intervals.
    return ((inter / union >= 0.30) & (inter / length[:, None] >= 0.50)).astype(np.float64)


_STEP_UTILITY = _utility(_T.astype(np.float64))


def interval_probabilities(logits):
    """Normalize a finite [N,496] array on the candidate axis."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != N_SPANS or not np.isfinite(x).all():
        raise ValueError("M2 MBR requires finite logits [N,496].")
    mass = np.exp(x - x.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def locked_reference_probabilities(window_inside, cra_answer, questions):
    """Compute C2 and C4 on M1 inside means [N,56,6], not contrast/log-alpha.

    Column 0 is Q-only, columns 1..5 are Q+candidate, cra_answer is 0-based.
    C2 uses only the question strings for its interval distribution.
    Share the same locked span order and centering arithmetic as the replay.
    """
    U = np.asarray(window_inside, dtype=np.float64)
    c = np.asarray(cra_answer)
    n = len(questions)
    if U.shape != (n, N_BANK, 6) or not np.isfinite(U).all():
        raise ValueError("Expected finite M1 inside means [N,56,6].")
    if c.shape != (n,) or c.dtype.kind not in "iu" or np.any((c < 0) | (c >= 5)):
        raise ValueError("Expected N CRA answer indices in 0..4.")
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("C2/C4 require nonempty source question strings.")

    tokens = [set(re.findall(r"[a-z']+", q.lower())) for q in questions]
    begin = np.asarray([bool(w & _BEGIN) for w in tokens], dtype=bool)
    end = np.asarray([bool(w & _END) for w in tokens], dtype=bool) & ~begin
    prior = -16.0 * (
        begin.astype(float)[:, None] * _CENTER[None, :]
        + end.astype(float)[:, None] * (1.0 - _CENTER[None, :])
    )
    fields = np.einsum("kl,nlc->nck", _PINV, U)
    field = fields[np.arange(n), 1 + c]
    z = (field - field.mean(1, keepdims=True)) / np.maximum(
        field.std(1, keepdims=True), 1e-9
    )
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(z, axis=1)], axis=1)
    t_g = (cs[:, _E] - cs[:, _S]) / np.sqrt(_E - _S)[None, :]
    return {
        "c2": interval_probabilities(prior),
        "c4": interval_probabilities(t_g + prior),
    }


def select(probabilities, timestamps=None):
    """Return [N,2] indices in stable, start-major order, plus winning utility.

    A shared timestamp vector can be supplied for questions about one video.
    None selects the original step-utility replay (not the main seconds arm).
    Equal floating-point maxima use the first candidate, as in the C2/C4 lock.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        p.ndim != 2 or p.shape[1] != N_SPANS or not np.isfinite(p).all()
        or np.any(p < 0) or not np.allclose(p.sum(1), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("M2 MBR expects normalized probabilities [N,496].")
    if timestamps is None:
        # Locked replay: do not change the step-domain utility.
        utility = _STEP_UTILITY
        scores = p @ utility.T
        choice = scores.argmax(axis=1)
        return SPAN_BOUNDS[choice], scores[np.arange(len(choice)), choice]

    t = validate_timestamps(timestamps)
    if np.all(np.diff(t) > 0):
        # Exactly the original 496-span seconds utility/MBR for strict times.
        scores = p @ _utility(t).T
        choice = scores.argmax(axis=1)
        return SPAN_BOUNDS[choice], scores[np.arange(len(choice)), choice]

    # The ONLY changed geometry: duplicated sample centers create some
    # zero-second intervals. Keep original indices/order and exclude *only*
    # those impossible positive-duration decisions and hypotheses. No fallback.
    valid = t[_E - 1] > t[_S]
    if not np.any(valid):
        raise ValueError("No positive-duration span among the 496 intervals.")
    bounds = SPAN_BOUNDS[valid]
    lo, hi = t[bounds[:, 0]], t[bounds[:, 1] - 1]
    lengths = hi - lo
    intersection = np.maximum(
        0.0, np.minimum(hi[:, None], hi[None, :])
        - np.maximum(lo[:, None], lo[None, :]),
    )
    union = lengths[:, None] + lengths[None, :] - intersection
    utility = ((intersection / union >= 0.30)
               & (intersection / lengths[:, None] >= 0.50)).astype(np.float64)
    pmass = p[:, valid]
    denom = pmass.sum(axis=1, keepdims=True)
    if not np.all(denom > 0):
        raise RuntimeError("All C4 probability mass fell on zero-duration spans.")
    scores = (pmass / denom) @ utility.T
    choice = scores.argmax(axis=1)
    return bounds[choice], scores[np.arange(len(choice)), choice]
