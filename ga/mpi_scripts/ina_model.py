import os
import ctypes
import pandas as pd
import numpy as np


class InaModel:

    def __init__(self, filename_so):

        filename_so_abs = os.path.abspath(filename_so)
        ctypes_obj = ctypes.CDLL(filename_so_abs)
        ctypes_obj.run.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'
            ),
            ctypes.c_int,
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'
            )
        ]
        ctypes_obj.run.restype = ctypes.c_int

        self._run = ctypes_obj.run
        self._status = None

    @property
    def status(self):
        return self._status

    def run(self, A, S, C, df_protocol, df_initial_state_protocol,
            n_sections,
            return_algebraic=False, **kwargs):

        t = df_protocol.t.values
        v_all = df_protocol.v.values

        t0 = df_initial_state_protocol.t.values
        v0 = df_initial_state_protocol.v.values

        output_len = len(t)
        initial_state_len = len(t0)

        initial_state_S = np.zeros((initial_state_len, len(S)))
        initial_state_A = np.zeros((initial_state_len, len(A)))

        self._run(
            S.values.copy(),
            C.values.copy(),
            t0,
            v0,
            initial_state_len,
            initial_state_S,
            initial_state_A
            )

        S_output = np.zeros((output_len, len(S)))
        A_output = np.zeros((output_len, len(A)))
        
        split_indices = np.linspace(0, output_len, n_sections + 1).astype(int)
        null_start, null_end = split_indices[0], split_indices[1]
        len_one_step = null_end - null_start
        t1 = t[null_start:null_end]
        S0 = initial_state_S[-1].copy()

        for k in range(n_sections):
            start, end = split_indices[k], split_indices[k + 1]
            v = v_all[start:end]
            self._status = self._run(
                S0,
                C.values.copy(),
                t1,
                v,
                len_one_step,
                S_output[start:end], A_output[start:end]
                )
        df_A = pd.DataFrame(A_output, columns=A.index)
        df_S = pd.DataFrame(S_output, columns=S.index)
        df_S['grad'] = np.gradient(df_S.I_out)
        
        if return_algebraic:
            return df_S, df_A
        return df_S
