#include "FacAna.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace mulsta {
	// =================================== Factor Analysis
	FacAna::FacAna(std::vector<std::vector<double>>& sample_data) {
		size_t n_samples = sample_data.size();
		size_t n_features = sample_data[0].size();
		sample_data_matrix.resize(n_samples, n_features);
		sample_data_cov.resize(n_features, n_features);
		correlation_matrix.resize(n_features, n_features);
		
		for (size_t i = 0; i < n_samples; i++) {
			for (size_t j = 0; j < n_features; j++) {
				sample_data_matrix(i, j) = sample_data[i][j];
			}
		}
	}

	void FacAna::factor_analysis() {
		normlize();
		compute_correlation();
		extract_factors();
		compute_loadings();
		compute_communalities();
		
		std::cout << "========== Factor Analysis Results (Based on PCA) ==========" << std::endl;
		std::cout << "Number of Variables: " << sample_data_matrix.cols() << std::endl;
		std::cout << "Number of Samples: " << sample_data_matrix.rows() << std::endl;
		std::cout << "Number of Factors: " << n_factors << std::endl;
		std::cout << "Cumulative Contribution Rate: " << std::fixed << std::setprecision(4) 
		          << cumulative_contribution * 100.0 << "%" << std::endl;
		std::cout << std::endl;
		
		std::cout << "Eigenvalues of Correlation Matrix:" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		double total_variance = eigenvalues.sum();
		for (int i = 0; i < eigenvalues.size(); ++i) {
			double contribution = eigenvalues(i) / total_variance;
			std::cout << "Factor " << (i + 1) << ": Eigenvalue = " << std::fixed << std::setprecision(6) 
			          << eigenvalues(i) << ", Contribution = " << std::fixed << std::setprecision(4) 
			          << contribution * 100.0 << "%" << std::endl;
		}
		std::cout << std::endl;
		
		std::cout << "Factor Loadings Matrix:" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		std::cout << std::setw(12) << "Variable";
		for (int j = 0; j < n_factors; ++j) {
			std::cout << std::setw(12) << "Factor" + std::to_string(j + 1);
		}
		std::cout << std::setw(12) << "Communality" << std::setw(12) << "Specific Var" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		
		for (int i = 0; i < factor_loadings.rows(); ++i) {
			std::cout << std::setw(12) << "Var" + std::to_string(i + 1);
			for (int j = 0; j < n_factors; ++j) {
				std::cout << std::setw(12) << std::fixed << std::setprecision(4) << factor_loadings(i, j);
			}
			std::cout << std::setw(12) << std::fixed << std::setprecision(4) << communalities(i);
			std::cout << std::setw(12) << std::fixed << std::setprecision(4) << specific_variances(i) << std::endl;
		}
		std::cout << std::string(80, '=') << std::endl;
	}

	void FacAna::normlize() {
		int col = sample_data_matrix.cols();
		for (int i = 0; i < col; i++) {
			double mean = 0.0;
			compute_mean(i, mean);
			double variance = 0.0;
			compute_varience(i, variance);
			double std_dev = std::sqrt(variance);
			if (std_dev > 1e-10) {
				sample_data_matrix.col(i) = (sample_data_matrix.col(i).array() - mean) / std_dev;
			}
		}
	}

	void FacAna::compute_mean(int& i, double& mean) {
		mean = sample_data_matrix.col(i).mean();
	}

	void FacAna::compute_varience(int& i, double& varience) {
		double mean = 0.0;
		compute_mean(i, mean);
		Eigen::VectorXd col_vec = sample_data_matrix.col(i);
		varience = (col_vec.array() - mean).square().mean();
	}

	void FacAna::compute_correlation() {
		Eigen::Index n_samples = sample_data_matrix.rows();
		Eigen::Index n_features = sample_data_matrix.cols();
		correlation_matrix.resize(n_features, n_features);
		correlation_matrix = (sample_data_matrix.transpose() * sample_data_matrix) / (n_samples - 1.0);
	}

	void FacAna::extract_factors() {
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(correlation_matrix);
		
		if (solver.info() != Eigen::Success) {
			n_factors = 0;
			cumulative_contribution = 0.0;
			return;
		}
		
		eigenvalues = solver.eigenvalues();
		eigenvectors = solver.eigenvectors();
		
		std::vector<std::pair<double, int>> eigen_pairs;
		for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
			eigen_pairs.push_back(std::make_pair(eigenvalues(i), i));
		}
		std::sort(eigen_pairs.begin(), eigen_pairs.end(), 
		          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
		              return a.first > b.first;
		          });
		
		Eigen::VectorXd sorted_eigenvalues(eigenvalues.size());
		Eigen::MatrixXd sorted_eigenvectors(eigenvectors.rows(), eigenvectors.cols());
		for (size_t i = 0; i < eigen_pairs.size(); ++i) {
			sorted_eigenvalues(i) = eigen_pairs[i].first;
			sorted_eigenvectors.col(i) = eigenvectors.col(eigen_pairs[i].second);
		}
		eigenvalues = sorted_eigenvalues;
		eigenvectors = sorted_eigenvectors;
		
		double total_variance = eigenvalues.sum();
		double cumulative_variance = 0.0;
		n_factors = 0;
		
		for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
			if (eigenvalues(i) > 1.0) {
				cumulative_variance += eigenvalues(i);
				n_factors++;
			} else {
				if (cumulative_variance / total_variance < 0.85) {
					cumulative_variance += eigenvalues(i);
					n_factors++;
				} else {
					break;
				}
			}
		}
		
		if (n_factors == 0) {
			n_factors = 1;
			cumulative_variance = eigenvalues(0);
		}
		
		cumulative_contribution = cumulative_variance / total_variance;
	}

	void FacAna::compute_loadings() {
		factor_loadings.resize(eigenvectors.rows(), n_factors);
		
		for (int j = 0; j < n_factors; ++j) {
			double sqrt_eigenvalue = std::sqrt(eigenvalues(j));
			factor_loadings.col(j) = eigenvectors.col(j) * sqrt_eigenvalue;
		}
	}

	void FacAna::compute_communalities() {
		communalities.resize(factor_loadings.rows());
		specific_variances.resize(factor_loadings.rows());
		
		for (int i = 0; i < factor_loadings.rows(); ++i) {
			double communality = 0.0;
			for (int j = 0; j < n_factors; ++j) {
				communality += factor_loadings(i, j) * factor_loadings(i, j);
			}
			communalities(i) = communality;
			specific_variances(i) = 1.0 - communality;
		}
	}
}
