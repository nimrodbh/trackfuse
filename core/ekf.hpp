#pragma once

#include "mat.hpp"

namespace tf {

// State: [x, y, heading, speed]. Controls (measured by an IMU): longitudinal
// acceleration a and yaw rate w. Motion model (unicycle):
//   x' = x + v cos(h) dt,  y' = y + v sin(h) dt,  h' = h + w dt,  v' = v + a dt
struct Params {
    double sigma_a = 0.2;    // std of the acceleration input, m/s^2
    double sigma_w = 0.01;   // std of the yaw-rate input, rad/s
    double gate_chi2 = 0.0;  // Mahalanobis gate (chi-square, 2 dof); 0 disables gating
};

inline double wrap_angle(double a) {
    const double two_pi = 6.283185307179586476925286766559;
    a = std::fmod(a + 3.14159265358979323846, two_pi);
    if (a < 0) {
        a += two_pi;
    }
    return a - 3.14159265358979323846;
}

class Ekf {
public:
    Ekf(const double x0[4], const double p0_diag[4], const Params& params);

    // Propagate the state by dt using the measured controls.
    void predict(double dt, double accel, double yaw_rate);

    // Position fix (e.g. GPS). Returns false if the gate rejected it.
    bool update_gps(double zx, double zy, double sigma);

    // Range and bearing to a landmark at a known position. Bearing is relative to the heading.
    bool update_landmark(double lx, double ly, double range, double bearing, double sigma_r,
                         double sigma_b);

    const Mat<4, 1>& state() const { return x_; }
    const Mat<4, 4>& cov() const { return p_; }
    long rejected() const { return rejected_; }

private:
    bool update2(const Mat<2, 4>& h, const Mat<2, 1>& innovation, const Mat<2, 2>& r);

    Mat<4, 1> x_;
    Mat<4, 4> p_;
    Params params_;
    long rejected_ = 0;
};

}  // namespace tf
