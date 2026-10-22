// C interface used by the Python bindings (ctypes). One call replays a whole recorded run.
#include <cstdint>

#include "ekf.hpp"

extern "C" {

// Event types for tf_run. Each event has 6 doubles in `vals`:
//   TF_PREDICT  : dt, accel, yaw_rate, -, -, -
//   TF_GPS      : zx, zy, sigma, -, -, -
//   TF_LANDMARK : lx, ly, range, bearing, sigma_r, sigma_b
enum { TF_PREDICT = 0, TF_GPS = 1, TF_LANDMARK = 2 };

// Replays n events. After every event the state (4 doubles) and covariance (16 doubles, row
// major) are written to out_state[i*4...] and out_cov[i*16...] (out_cov may be NULL).
// Returns the number of rejected measurements, or -1 on an invalid argument.
long tf_run(int n, const int32_t* types, const double* vals, const double* x0, const double* p0_diag,
            double sigma_a, double sigma_w, double gate_chi2, double* out_state, double* out_cov) {
    if (n < 0 || types == nullptr || vals == nullptr || x0 == nullptr || p0_diag == nullptr ||
        out_state == nullptr) {
        return -1;
    }
    tf::Params params;
    params.sigma_a = sigma_a;
    params.sigma_w = sigma_w;
    params.gate_chi2 = gate_chi2;
    tf::Ekf ekf(x0, p0_diag, params);

    for (int i = 0; i < n; i++) {
        const double* v = vals + 6 * i;
        switch (types[i]) {
            case TF_PREDICT:
                ekf.predict(v[0], v[1], v[2]);
                break;
            case TF_GPS:
                ekf.update_gps(v[0], v[1], v[2]);
                break;
            case TF_LANDMARK:
                ekf.update_landmark(v[0], v[1], v[2], v[3], v[4], v[5]);
                break;
            default:
                return -1;
        }
        for (int k = 0; k < 4; k++) {
            out_state[4 * i + k] = ekf.state()(k, 0);
        }
        if (out_cov != nullptr) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    out_cov[16 * i + 4 * r + c] = ekf.cov()(r, c);
                }
            }
        }
    }
    return ekf.rejected();
}

}  // extern "C"
