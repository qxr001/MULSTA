#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <string>
#include <utility>

namespace mulsta {
	double get_t_critical(int df, double alpha);
	std::string get_significance_level(double t_stat, double t_005, double t_001);
	class RegAna {
	public:
		RegAna(std::vector<std::vector<double>>& sample_data);
		~RegAna() {}

	public:
		void regressiona_analysis();
		double point_prediction(std::vector<double>& x_values);
		std::pair<double, double> interval_prediction(std::vector<double>& x_values, double alpha = 0.05);

	protected:
		void compute_coefficients();
		void compute_sums_of_squares();
		void compute_coefficient_tests();

	protected:
		Eigen::MatrixXd X;
		Eigen::VectorXd y;
		Eigen::VectorXd beta;
		Eigen::VectorXd y_pred;
		Eigen::VectorXd residuals;
		
		int n_samples;
		int n_features;
		
		double SST;
		double SSR;
		double SSE;
		double MSR;
		double MSE;
		double F_statistic;
		double y_mean;
		
		Eigen::VectorXd beta_std_errors;
		Eigen::VectorXd t_statistics;
		Eigen::MatrixXd XtX_inv;

	};

}
