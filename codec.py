"""An educational ternary format: separate byte-aligned groups, fp16 scales."""

import numpy as np

POWERS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def pack_trits(trits):
    """Pack each row independently, least-significant digit first; pad with zero trits."""
    trits = np.asarray(trits)
    if trits.ndim != 2 or not np.isin(trits, [-1, 0, 1]).all():
        raise ValueError("Expected rows of trits in {-1, 0, 1}")
    digits = (trits.astype(np.int16) + 1).astype(np.uint16)
    digits = np.pad(digits, ((0, 0), (0, (-digits.shape[1]) % 5)), constant_values=1)
    return (digits.reshape(len(digits), -1, 5) @ POWERS).astype(np.uint8)


def unpack_trits(payload, group_size):
    payload = np.asarray(payload)
    if (group_size < 1 or payload.ndim != 2 or payload.dtype != np.uint8
            or payload.shape[1] != (group_size + 4) // 5 or (payload > 242).any()):
        raise ValueError("Invalid ternary payload")
    digits = (payload.astype(np.uint16)[..., None] // POWERS) % 3
    trits = digits.reshape(len(payload), -1).astype(np.int8) - 1
    if np.any(trits[:, group_size:] != 0):
        raise ValueError("Nonzero padding trit")
    return trits[:, :group_size]


def quantize(weight, group_size):
    weight = np.asarray(weight, dtype=np.float32)
    if (weight.ndim != 2 or group_size < 1 or weight.shape[1] % group_size
            or not np.isfinite(weight).all()):
        raise ValueError("Expected finite matrix with columns divisible by group size")
    groups = weight.reshape(-1, group_size)
    # Use the scale that will actually be stored, including its fp16 rounding.
    scales = np.mean(np.abs(groups), axis=1).astype("<f2")
    if not np.isfinite(scales).all():
        raise ValueError("Scale overflows fp16")
    safe_scales = np.where(scales > 0, scales.astype(np.float32), 1)
    trits = np.clip(np.rint(groups / safe_scales[:, None]), -1, 1).astype(np.int8)
    trits[scales == 0] = 0
    return trits, scales


def decode(payload, scales, shape, group_size):
    if len(shape) != 2 or shape[1] % group_size:
        raise ValueError("Invalid tensor shape")
    if scales.dtype != np.dtype("<f2") or scales.shape != (np.prod(shape) // group_size,):
        raise ValueError("Invalid scale plane")
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("Invalid scale value")
    trits = unpack_trits(payload, group_size)
    if len(trits) != len(scales):
        raise ValueError("Payload and scale counts differ")
    return (trits.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(shape)
