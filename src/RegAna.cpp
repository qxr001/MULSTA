#include "RegAna.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace mulsta {
	// =================================== Regression Analysis
	RegAna::RegAna(std::vector<std::vector<double>>& sample_data) {
		size_t n_samples = sample_data.size();
		if (n_samples == 0) {
			n_features = 0;
			this->n_samples = 0;
			return;
		}
		
		size_t n_cols = sample_data[0].size();
		if (n_cols < 2) {
			n_features = 0;
			this->n_samples = 0;
			return;
		}
		
		this->n_samples = static_cast<int>(n_samples);
		n_features = static_cast<int>(n_cols - 1); 
		
		X.resize(n_samples, n_features + 1);
		y.resize(n_samples);
		
		for (size_t i = 0; i < n_samples; i++) {
			X(i, 0) = 1.0;
			for (size_t j = 0; j < n_features; j++) {
				X(i, j + 1) = sample_data[i][j + 1];
			}
			y(i) = sample_data[i][0];
		}
		
		beta.resize(n_features + 1);
		y_pred.resize(n_samples);
		residuals.resize(n_samples);
		beta_std_errors.resize(n_features + 1);
		t_statistics.resize(n_features + 1);
		XtX_inv.resize(n_features + 1, n_features + 1);
	}

	void RegAna::regressiona_analysis() {
		if (n_samples == 0 || n_features == 0) {
			std::cout << "Error: Invalid data for regression analysis." << std::endl;
			return;
		}
		
		compute_coefficients();
		
		compute_sums_of_squares();
		
		compute_coefficient_tests();
		
		std::cout << "========== Regression Analysis Results (Least Squares) ==========" << std::endl;
		std::cout << "Number of Samples: " << n_samples << std::endl;
		std::cout << "Number of Independent Variables: " << n_features << std::endl;
		std::cout << std::endl;
		
		std::cout << "Regression Coefficients and Significance Tests:" << std::endl;
		std::cout << std::string(100, '-') << std::endl;
		std::cout << std::setw(15) << "Coefficient" << std::setw(15) << "Value" 
		          << std::setw(15) << "Std Error" << std::setw(15) << "t-Statistic" 
		          << std::setw(20) << "Significance" << std::endl;
		std::cout << std::string(100, '-') << std::endl;
		
		int df = n_samples - n_features - 1;
		double t_critical_005 = get_t_critical(df, 0.05);
		double t_critical_001 = get_t_critical(df, 0.01);
		
		std::string sig_intercept = get_significance_level(t_statistics(0), t_critical_005, t_critical_001);
		std::cout << std::setw(15) << "Intercept" 
		          << std::setw(15) << std::fixed << std::setprecision(6) << beta(0)
		          << std::setw(15) << std::fixed << std::setprecision(6) << beta_std_errors(0)
		          << std::setw(15) << std::fixed << std::setprecision(6) << t_statistics(0)
		          << std::setw(20) << sig_intercept << std::endl;
		
		for (int i = 0; i < n_features; i++) {
			std::string sig = get_significance_level(t_statistics(i + 1), t_critical_005, t_critical_001);
			std::cout << std::setw(15) << "X" + std::to_string(i + 1) 
			          << std::setw(15) << std::fixed << std::setprecision(6) << beta(i + 1)
			          << std::setw(15) << std::fixed << std::setprecision(6) << beta_std_errors(i + 1)
			          << std::setw(15) << std::fixed << std::setprecision(6) << t_statistics(i + 1)
			          << std::setw(20) << sig << std::endl;
		}
		std::cout << std::endl;
		std::cout << "Note: Significance levels: *** p<0.01, ** p<0.05, * p<0.10, ns not significant" << std::endl;
		std::cout << "Degrees of Freedom: " << df << std::endl;
		std::cout << std::endl;
		
		std::cout << "Analysis of Variance (ANOVA):" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		std::cout << std::setw(20) << "Source" << std::setw(20) << "Sum of Squares" 
		          << std::setw(20) << "Mean Square" << std::setw(20) << "F-Statistic" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		std::cout << std::setw(20) << "Regression" << std::setw(20) << std::fixed << std::setprecision(6) << SSR
		          << std::setw(20) << std::fixed << std::setprecision(6) << MSR
		          << std::setw(20) << std::fixed << std::setprecision(6) << F_statistic << std::endl;
		std::cout << std::setw(20) << "Residual" << std::setw(20) << std::fixed << std::setprecision(6) << SSE
		          << std::setw(20) << std::fixed << std::setprecision(6) << MSE
		          << std::setw(20) << "-" << std::endl;
		std::cout << std::setw(20) << "Total" << std::setw(20) << std::fixed << std::setprecision(6) << SST
		          << std::setw(20) << "-" << std::setw(20) << "-" << std::endl;
		std::cout << std::endl;
		
		std::cout << "Summary Statistics:" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		std::cout << "Total Sum of Squares (SST): " << std::fixed << std::setprecision(6) << SST << std::endl;
		std::cout << "Regression Sum of Squares (SSR): " << std::fixed << std::setprecision(6) << SSR << std::endl;
		std::cout << "Residual Sum of Squares (SSE): " << std::fixed << std::setprecision(6) << SSE << std::endl;
		std::cout << "Mean Square Regression (MSR): " << std::fixed << std::setprecision(6) << MSR << std::endl;
		std::cout << "Mean Square Error (MSE): " << std::fixed << std::setprecision(6) << MSE << std::endl;
		std::cout << "F-Statistic (F = MSR/MSE): " << std::fixed << std::setprecision(6) << F_statistic << std::endl;
		std::cout << std::string(80, '=') << std::endl;
		std::cout << std::endl;
	}

	void RegAna::compute_coefficients() {
		Eigen::MatrixXd XtX = X.transpose() * X;
		Eigen::VectorXd Xty = X.transpose() * y;
		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
		beta = qr.solve(y);
		
		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_XtX(XtX);
		XtX_inv = qr_XtX.solve(Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols()));
		
		y_pred = X * beta;
		residuals = y - y_pred;
		y_mean = y.mean();
	}

	void RegAna::compute_sums_of_squares() {
		SST = (y.array() - y_mean).square().sum();
		SSR = (y_pred.array() - y_mean).square().sum();
		SSE = residuals.squaredNorm();
		MSR = SSR / n_features;
		
		int degrees_of_freedom_error = n_samples - n_features - 1;
		if (degrees_of_freedom_error > 0) {
			MSE = SSE / degrees_of_freedom_error;
		} else {
			MSE = 0.0;
		}

		if (MSE > 1e-10) {
			F_statistic = MSR / MSE;
		} else {
			F_statistic = 0.0;
		}
	}

	void RegAna::compute_coefficient_tests() {
		for (int i = 0; i < n_features + 1; i++) {
			if (MSE > 1e-10 && XtX_inv(i, i) > 0) {
				beta_std_errors(i) = std::sqrt(MSE * XtX_inv(i, i));
			} else {
				beta_std_errors(i) = 0.0;
			}
		}
		
		for (int i = 0; i < n_features + 1; i++) {
			if (beta_std_errors(i) > 1e-10) {
				t_statistics(i) = beta(i) / beta_std_errors(i);
			} else {
				t_statistics(i) = 0.0;
			}
		}
	}

	double RegAna::point_prediction(std::vector<double>& x_values) {
		if (x_values.size() != n_features) {
			std::cerr << "Error: Number of independent variables does not match." << std::endl;
			return 0.0;
		}
		
		Eigen::VectorXd x0(n_features + 1);
		x0(0) = 1.0;
		for (int i = 0; i < n_features; i++) {
			x0(i + 1) = x_values[i];
		}
		
		double y_hat = x0.transpose() * beta;
		return y_hat;
	}

	std::pair<double, double> RegAna::interval_prediction(std::vector<double>& x_values, double alpha) {
		if (x_values.size() != n_features) {
			std::cerr << "Error: Number of independent variables does not match." << std::endl;
			return std::make_pair(0.0, 0.0);
		}
		
		Eigen::VectorXd x0(n_features + 1);
		x0(0) = 1.0;
		for (int i = 0; i < n_features; i++) {
			x0(i + 1) = x_values[i];
		}
		
		double y_hat = x0.transpose() * beta;
		
		double x0_XtXinv_x0 = x0.transpose() * XtX_inv * x0;
		double se_y_hat = std::sqrt(MSE * (1.0 + x0_XtXinv_x0));
		
		int df = n_samples - n_features - 1;
		double t_critical = get_t_critical(df, alpha);
		
		double margin = t_critical * se_y_hat;
		double lower_bound = y_hat - margin;
		double upper_bound = y_hat + margin;
		
		return std::make_pair(lower_bound, upper_bound);
	}

	double get_t_critical(int df, double alpha) {
		if (df <= 0) return 0.0;
		
		double z = 0.0;
		if (alpha == 0.10) z = 1.645;
		else if (alpha == 0.05) z = 1.96;
		else if (alpha == 0.01) z = 2.576;
		else {
			z = 1.96;
		}
		
		if (df < 30) {
			double correction = 1.0 + 1.0 / (2.0 * df);
			z *= correction;
		}
		
		double z_alpha_2 = 0.0;
		if (alpha == 0.10) z_alpha_2 = 1.645;
		else if (alpha == 0.05) z_alpha_2 = 1.96;
		else if (alpha == 0.01) z_alpha_2 = 2.576;
		else z_alpha_2 = 1.96;
		
		double t_critical = z_alpha_2 * (1.0 + z_alpha_2 * z_alpha_2 / (4.0 * df));
		
		if (df <= 5) {
			if (alpha == 0.05) return 2.571;
			if (alpha == 0.01) return 4.032;
		}
		if (df <= 10) {
			if (alpha == 0.05) return 2.228;
			if (alpha == 0.01) return 3.169;
		}
		if (df <= 20) {
			if (alpha == 0.05) return 2.086;
			if (alpha == 0.01) return 2.845;
		}
		
		return t_critical;
	}

	std::string get_significance_level(double t_stat, double t_005, double t_001) {
		double abs_t = std::abs(t_stat);
		if (abs_t >= t_001) {
			return "*** (p<0.01)";
		} else if (abs_t >= t_005) {
			return "** (p<0.05)";
		} else if (abs_t >= 1.645) {
			return "* (p<0.10)";
		} else {
			return "ns (not significant)";
		}
	}
}
