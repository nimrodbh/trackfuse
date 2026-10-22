#pragma once
// Minimal fixed-size matrix: enough linear algebra for a 4-state Kalman filter,
// with sizes checked at compile time and no heap allocation.

#include <array>
#include <cmath>

namespace tf {

template <int R, int C>
struct Mat {
    std::array<double, R * C> a{};

    double& operator()(int i, int j) { return a[static_cast<std::size_t>(i * C + j)]; }
    double operator()(int i, int j) const { return a[static_cast<std::size_t>(i * C + j)]; }

    static Mat identity() {
        static_assert(R == C, "identity needs a square matrix");
        Mat m;
        for (int i = 0; i < R; i++) {
            m(i, i) = 1.0;
        }
        return m;
    }
};

template <int R, int K, int C>
Mat<R, C> operator*(const Mat<R, K>& A, const Mat<K, C>& B) {
    Mat<R, C> out;
    for (int i = 0; i < R; i++) {
        for (int k = 0; k < K; k++) {
            const double aik = A(i, k);
            for (int j = 0; j < C; j++) {
                out(i, j) += aik * B(k, j);
            }
        }
    }
    return out;
}

template <int R, int C>
Mat<R, C> operator+(const Mat<R, C>& A, const Mat<R, C>& B) {
    Mat<R, C> out;
    for (std::size_t i = 0; i < out.a.size(); i++) {
        out.a[i] = A.a[i] + B.a[i];
    }
    return out;
}

template <int R, int C>
Mat<R, C> operator-(const Mat<R, C>& A, const Mat<R, C>& B) {
    Mat<R, C> out;
    for (std::size_t i = 0; i < out.a.size(); i++) {
        out.a[i] = A.a[i] - B.a[i];
    }
    return out;
}

template <int R, int C>
Mat<C, R> transpose(const Mat<R, C>& A) {
    Mat<C, R> out;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            out(j, i) = A(i, j);
        }
    }
    return out;
}

// Closed-form 2x2 inverse. Returns false (and leaves `out` zero) if the matrix is singular.
inline bool inv2(const Mat<2, 2>& m, Mat<2, 2>& out) {
    const double det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    if (!(std::fabs(det) > 1e-300)) {
        return false;
    }
    out(0, 0) = m(1, 1) / det;
    out(0, 1) = -m(0, 1) / det;
    out(1, 0) = -m(1, 0) / det;
    out(1, 1) = m(0, 0) / det;
    return true;
}

}  // namespace tf
