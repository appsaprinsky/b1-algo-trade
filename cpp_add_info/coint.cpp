#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

// Function to calculate the mean of a vector
double mean(const std::vector<double>& vec) {
    double sum = 0.0;
    for (const auto& val : vec) sum += val;
    return sum / vec.size();
}

// Function to calculate variance
double variance(const std::vector<double>& vec) {
    double mu = mean(vec);
    double var = 0.0;
    for (const auto& val : vec) var += (val - mu) * (val - mu);
    return var / vec.size();
}

// Function to perform linear regression
Eigen::VectorXd linearRegression(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    Eigen::MatrixXd X(n, 2);  // Design matrix
    Eigen::VectorXd Y(n);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = 1.0; // Intercept
        X(i, 1) = x[i];
        Y(i) = y[i];
    }

    // Beta = (X'X)^-1 X'Y
    Eigen::VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * Y;
    return beta;
}

// Function to calculate residuals
std::vector<double> calculateResiduals(const std::vector<double>& x, const std::vector<double>& y, const Eigen::VectorXd& beta) {
    std::vector<double> residuals;
    for (size_t i = 0; i < x.size(); ++i) {
        double predicted = beta(0) + beta(1) * x[i];
        residuals.push_back(y[i] - predicted);
    }
    return residuals;
}

// Simple ADF test (basic implementation)
bool adfTest(const std::vector<double>& residuals) {
    int n = residuals.size();
    std::vector<double> diffRes;
    for (size_t i = 1; i < n; ++i) {
        diffRes.push_back(residuals[i] - residuals[i - 1]);
    }

    // Regression of diffRes on lagged residuals
    std::vector<double> lagRes(residuals.begin(), residuals.end() - 1);
    Eigen::VectorXd beta = linearRegression(lagRes, diffRes);

    // Test statistic
    double t_stat = beta(1) / std::sqrt(variance(diffRes) / variance(lagRes));
    std::cout << "ADF Test Statistic: " << t_stat << std::endl;

    // Compare with critical value (-2.9 for alpha=0.05, simplified)
    return t_stat < -2.9;
}

int main() {
    // Example data (two time series)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> y = {1.1, 2.1, 3.0, 4.1, 5.0, 5.9, 7.1, 8.2};

    // Step 1: Linear regression of y on x
    Eigen::VectorXd beta = linearRegression(x, y);

    // Step 2: Calculate residuals
    std::vector<double> residuals = calculateResiduals(x, y, beta);

    // Step 3: Apply ADF test on residuals
    bool isCointegrated = adfTest(residuals);

    if (isCointegrated) {
        std::cout << "The time series are cointegrated." << std::endl;
    } else {
        std::cout << "The time series are not cointegrated." << std::endl;
    }

    return 0;
}
