# Inspired from https://github.com/z2e2/fastddtw/blob/master/_fastddtw.py
import numpy as np


## reference: https://github.com/slaypni/fastdtw
## adapted from https://github.com/slaypni/fastdtw/blob/master/fastdtw/fastdtw.__dtw)
def est_derivatives(sig):
    '''
    Computing drivative differences between dx and dy
    Arguments:
           sig -- signal, numpy array of shape ( n,  )
    Result:
         d_sig -- estimated derivatives of input signal, numpy array of shape ( n-2,  )
    '''
    assert len(sig) >= 3, '''The length of your signal should be
                           greater than 3 to implement DDTW.'''
    if type(sig) != np.ndarray:
        sig = np.array(sig)
    d_0 = sig[:-2]
    d_1 = sig[1:-1]
    d_2 = sig[2:]
    d_sig = ((d_1 - d_0) + (d_2 - d_0) / 2) / 2
    return d_sig


def generate_window(len_d_sig1, len_d_sig2, K):
    '''
    generate reduced search space
    Arguments:
           len_d_sig1 -- length of signal 1, scalar
           len_d_sig2 -- length of signal 2, scalar
                    K -- window size is 2 * K, scalar
    Result:
               window -- reduced search space, a python generator
    '''
    for i in range(len_d_sig1):
        lb = i - K
        ub = i + K
        if lb < 0 and ub < len_d_sig2:
            for j in range(ub):
                yield (i + 1, j + 1)
        elif lb >= 0 and ub < len_d_sig2:
            for j in range(lb, ub):
                yield (i + 1, j + 1)
        elif lb < 0 and ub >= len_d_sig2:
            for j in range(len_d_sig2):
                yield (i + 1, j + 1)
        elif lb >= 0 and ub >= len_d_sig2:
            for j in range(lb, len_d_sig2):
                yield (i + 1, j + 1)


def _window_bounds(row, len_d_sig2, K):
    '''Column range ``[start, stop)`` that ``generate_window`` emits for ``row``.

    All four branches above reduce to the same half-open band; keeping it as one
    expression lets the dynamic program walk a row at a time.
    '''
    start = row - K
    if start < 0:
        start = 0
    stop = row + K
    if stop > len_d_sig2:
        stop = len_d_sig2
    return start, stop


def fast_ddtw(signal_1, signal_2, K=10, debug: bool = True):
    '''
    Arguments:
        signal_1 -- first time series, numpy array of shape ( n1,  )
        signal_2 -- second time series, numpy array of shape ( n2,  )
    Results:
        ddtw -- distance matrix, numpy array of shape ( n1 - 2, n2 - 2 )
        ddtw_traceback -- traceback matrix, numpy array of shape ( n1 - 2, n2 - 2 )
    '''

    d_sig1 = est_derivatives(signal_1)
    d_sig2 = est_derivatives(signal_2)

    len_d_sig1, len_d_sig2 = len(d_sig1), len(d_sig2)
    if K > abs(len_d_sig1 - len_d_sig2):
        pass
    else:
        K = 2 * abs(len_d_sig1 - len_d_sig2)
        if debug:
            print('input K is not a good choice... selected K = ', K, 'instead.')

    # The search space is a band of contiguous columns per row, so the table can
    # be kept as one list per row instead of a dict keyed by (i, j) tuples.  The
    # three candidates are compared as sums, in the same order the original
    # ``min(...)`` used, so ties pick the same predecessor and the traceback is
    # unchanged.
    x = d_sig1.tolist()
    y = d_sig2.tolist()
    inf = float('inf')
    up, left_code, diag = 0, 1, 2

    dirs = bytearray()
    offsets = [0] * len_d_sig1
    starts = [0] * len_d_sig1

    # Virtual row -1 holding only the origin, the original's ``D[0, 0] = 0``.
    prev = [0.0]
    prev_s = -1
    prev_e = -1

    cur = []
    s = 0
    e = -1
    for row in range(len_d_sig1):
        start, stop = _window_bounds(row, len_d_sig2, K)
        s = start
        e = stop - 1
        width = stop - start
        starts[row] = s
        offsets[row] = len(dirs)
        if width <= 0:
            # Unreachable row: the original leaves every cell at inf.
            prev = []
            prev_s = 0
            prev_e = -1
            cur = []
            continue

        # Previous-row values for columns [s-1, e], padded with inf so the inner
        # loop needs no bounds checks.
        pv = [inf] * (width + 1)
        lo = s - 1
        if lo < prev_s:
            lo = prev_s
        hi = e
        if hi > prev_e:
            hi = prev_e
        if lo <= hi:
            pv[lo - s + 1:hi - s + 2] = prev[lo - prev_s:hi - prev_s + 1]

        xi = x[row]
        ys = y[s:stop]
        cur = [0.0] * width
        row_dirs = bytearray(width)
        best = inf
        for k in range(width):
            dt = xi - ys[k]
            if dt < 0.0:
                dt = -dt
            a_up = pv[k + 1] + dt
            a_left = best + dt
            a_diag = pv[k] + dt
            if a_up <= a_left:
                if a_up <= a_diag:
                    best = a_up
                    row_dirs[k] = up
                else:
                    best = a_diag
                    row_dirs[k] = diag
            elif a_left <= a_diag:
                best = a_left
                row_dirs[k] = left_code
            else:
                best = a_diag
                row_dirs[k] = diag
            cur[k] = best
        dirs += row_dirs

        prev = cur
        prev_s = s
        prev_e = e

    if not cur or e != len_d_sig2 - 1:
        # Matches the original reading a missing D[len_d_sig1, len_d_sig2].
        return (inf, [])
    distance = cur[-1]

    path = []
    i = len_d_sig1 - 1
    j = len_d_sig2 - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        code = dirs[offsets[i] + j - starts[i]]
        if code == up:
            i -= 1
        elif code == left_code:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return (distance, path)
