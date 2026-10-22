#include "ekf.hpp"

#include <cmath>

namespace tf {

Ekf::Ekf(const double x0[4], const double p0_diag[4], const Params& params) : params_(params) {
    for (int i = 0; i < 4; i++) {
        x_(i, 0) = x0[i];
        p_(i, i) = p0_diag[i];
    }
}

void Ekf::predict(double dt, double accel, double yaw_rate) {
    const double h = x_(2, 0);
    const double v = x_(3, 0);
    const double c = std::cos(h);
    const double s = std::sin(h);

    // Jacobian of the motion model with respect to the state.
    Mat<4, 4> f = Mat<4, 4>::identity();
    f(0, 2) = -v * s * dt;
    f(0, 3) = c * dt;
    f(1, 2) = v * c * dt;
    f(1, 3) = s * dt;

    // Jacobian with respect to the controls (a, w): the IMU noise is the process noise.
    Mat<4, 2> g;
    g(2, 1) = dt;
    g(3, 0) = dt;
    Mat<2, 2> qu;
    qu(0, 0) = params_.sigma_a * params_.sigma_a;
    qu(1, 1) = params_.sigma_w * params_.sigma_w;

    x_(0, 0) += v * c * dt;
    x_(1, 0) += v * s * dt;
    x_(2, 0) = wrap_angle(h + yaw_rate * dt);
    x_(3, 0) += accel * dt;

    p_ = f * p_ * transpose(f) + g * qu * transpose(g);
}

bool Ekf::update2(const Mat<2, 4>& h, const Mat<2, 1>& innovation, const Mat<2, 2>& r) {
    const Mat<4, 2> pht = p_ * transpose(h);
    const Mat<2, 2> s = h * pht + r;
    Mat<2, 2> s_inv;
    if (!inv2(s, s_inv)) {
        rejected_++;
        return false;
    }

    if (params_.gate_chi2 > 0.0) {
        const Mat<1, 2> yt = transpose(innovation);
        const double d2 = (yt * s_inv * innovation)(0, 0);
        if (d2 > params_.gate_chi2) {
            rejected_++;
            return false;
        }
    }

    const Mat<4, 2> k = pht * s_inv;
    x_ = x_ + k * innovation;
    x_(2, 0) = wrap_angle(x_(2, 0));

    // Joseph form keeps the covariance symmetric and positive semi-definite.
    const Mat<4, 4> i_kh = Mat<4, 4>::identity() - k * h;
    p_ = i_kh * p_ * transpose(i_kh) + k * r * transpose(k);
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            const double m = 0.5 * (p_(i, j) + p_(j, i));
            p_(i, j) = m;
            p_(j, i) = m;
        }
    }
    return true;
}

bool Ekf::update_gps(double zx, double zy, double sigma) {
    Mat<2, 4> h;
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;
    Mat<2, 1> y;
    y(0, 0) = zx - x_(0, 0);
    y(1, 0) = zy - x_(1, 0);
    Mat<2, 2> r;
    r(0, 0) = sigma * sigma;
    r(1, 1) = sigma * sigma;
    return update2(h, y, r);
}

bool Ekf::update_landmark(double lx, double ly, double range, double bearing, double sigma_r,
                          double sigma_b) {
    const double dx = lx - x_(0, 0);
    const double dy = ly - x_(1, 0);
    const double q = dx * dx + dy * dy;
    const double rho = std::sqrt(q);
    if (rho < 1e-6) {
        rejected_++;
        return false;
    }

    Mat<2, 4> h;
    h(0, 0) = -dx / rho;
    h(0, 1) = -dy / rho;
    h(1, 0) = dy / q;
    h(1, 1) = -dx / q;
    h(1, 2) = -1.0;

    Mat<2, 1> y;
    y(0, 0) = range - rho;
    y(1, 0) = wrap_angle(bearing - (std::atan2(dy, dx) - x_(2, 0)));

    Mat<2, 2> r;
    r(0, 0) = sigma_r * sigma_r;
    r(1, 1) = sigma_b * sigma_b;
    return update2(h, y, r);
}

}  // namespace tf
