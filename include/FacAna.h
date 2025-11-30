#pragma once
#include <vector>
#include <Eigen/Eigen>

namespace mulsta {
	class FacAna {
	public:
		FacAna(std::vector<std::vector<double>>& sample_data);
		~FacAna(){}

	public:
		void factor_analysis();

	protected:
		void normlize();
		void compute_mean(int& i, double& mean);
		void compute_varience(int& i, double& varience);
		void compute_correlation();
		void extract_factors();
		void compute_loadings();
		void compute_communalities();

	protected:
		Eigen::MatrixXd sample_data_matrix;
		Eigen::MatrixXd sample_data_cov;
		Eigen::MatrixXd correlation_matrix;
		Eigen::MatrixXd factor_loadings;
		Eigen::VectorXd eigenvalues;
		Eigen::MatrixXd eigenvectors;
		Eigen::VectorXd communalities;
		Eigen::VectorXd specific_variances;
		int n_factors;
		double cumulative_contribution;

	};

}

