#include <cassert>
#include <cmath>
#include <cstdio>

#include "ekf.hpp"

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

static void test_matrix_ops() {
    tf::Mat<2, 3> a;
    tf::Mat<3, 2> b;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            a(i, j) = i + 2 * j + 1;
            b(j, i) = 3 * i - j + 2;
        }
    }
    const tf::Mat<2, 2> c = a * b;
    // Hand-computed: a = [[1,3,5],[2,4,6]], b = [[2,5],[1,4],[0,3]]
    assert(near(c(0, 0), 1 * 2 + 3 * 1 + 5 * 0));
    assert(near(c(0, 1), 1 * 5 + 3 * 4 + 5 * 3));
    assert(near(c(1, 0), 2 * 2 + 4 * 1 + 6 * 0));
    assert(near(c(1, 1), 2 * 5 + 4 * 4 + 6 * 3));
    const tf::Mat<3, 2> at = tf::transpose(a);
    assert(near(at(2, 1), a(1, 2)));

    tf::Mat<2, 2> m, inv;
    m(0, 0) = 4;
    m(0, 1) = 7;
    m(1, 0) = 2;
    m(1, 1) = 6;
    assert(tf::inv2(m, inv));
    const tf::Mat<2, 2> id = m * inv;
    assert(near(id(0, 0), 1) && near(id(1, 1), 1) && near(id(0, 1), 0) && near(id(1, 0), 0));

    tf::Mat<2, 2> singular;  // all zeros
    tf::Mat<2, 2> out;
    assert(!tf::inv2(singular, out));
}

static void test_straight_line() {
    const double x0[4] = {0, 0, 0, 10};  // heading east at 10 m/s
    const double p0[4] = {1, 1, 0.01, 1};
    tf::Ekf ekf(x0, p0, tf::Params{});
    for (int i = 0; i < 100; i++) {
        ekf.predict(0.01, 0.0, 0.0);
    }
    assert(near(ekf.state()(0, 0), 10.0, 1e-9));  // 10 m/s for 1 s
    assert(near(ekf.state()(1, 0), 0.0, 1e-9));
    assert(ekf.cov()(0, 0) > p0[0]);  // uncertainty grows without measurements
}

static void test_update_shrinks_uncertainty_and_pulls_state() {
    const double x0[4] = {0, 0, 0, 0};
    const double p0[4] = {25, 25, 0.1, 1};
    tf::Ekf ekf(x0, p0, tf::Params{});
    const double before = ekf.cov()(0, 0);
    assert(ekf.update_gps(4.0, -2.0, 1.0));
    assert(ekf.cov()(0, 0) < before);
    assert(ekf.state()(0, 0) > 3.5 && ekf.state()(0, 0) < 4.0);  // close to the precise fix
    assert(ekf.state()(1, 0) < -1.5 && ekf.state()(1, 0) > -2.0);
    // symmetric
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            assert(near(ekf.cov()(i, j), ekf.cov()(j, i), 1e-12));
        }
    }
}

static void test_landmark_corrects_heading() {
    const double x0[4] = {0, 0, 0.3, 0};  // believes heading 0.3 rad
    const double p0[4] = {0.01, 0.01, 1.0, 0.01};
    tf::Ekf ekf(x0, p0, tf::Params{});
    // Landmark straight east at 20 m. True heading is 0, so true bearing is 0.
    assert(ekf.update_landmark(20, 0, 20.0, 0.0, 0.1, 0.01));
    assert(std::fabs(ekf.state()(2, 0)) < 0.05);  // heading pulled from 0.3 toward 0
}

static void test_gate_rejects_outlier() {
    const double x0[4] = {0, 0, 0, 0};
    const double p0[4] = {1, 1, 0.1, 1};
    tf::Params p;
    p.gate_chi2 = 9.21;  // 99% for 2 dof
    tf::Ekf ekf(x0, p0, p);
    assert(!ekf.update_gps(500.0, 500.0, 1.0));
    assert(ekf.rejected() == 1);
    assert(near(ekf.state()(0, 0), 0.0));  // untouched
    assert(ekf.update_gps(0.5, -0.5, 1.0));
}

static void test_angle_wrap() {
    assert(near(tf::wrap_angle(3.14159265358979323846 + 0.1), -3.14159265358979323846 + 0.1, 1e-12));
    assert(near(tf::wrap_angle(-7.0), -7.0 + 6.283185307179586, 1e-12));
    assert(near(tf::wrap_angle(0.5), 0.5, 1e-12));
}

int main() {
    test_matrix_ops();
    test_straight_line();
    test_update_shrinks_uncertainty_and_pulls_state();
    test_landmark_corrects_heading();
    test_gate_rejects_outlier();
    test_angle_wrap();
    std::puts("all C++ tests passed");
    return 0;
}
