#include "PCA.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <map>
#include <iostream>
#include <iomanip>
namespace mulsta {
	// =================================== PCA
	PCA::PCA(std::vector<int>& adaMath,
		std::vector<int>& proSta,
		std::vector<int>& linAlge,
		std::vector<int>& colEng,
		std::vector<int>& statt,
		std::vector<int>& python) {
		// do not use.
	}

	PCA::PCA(std::vector<std::vector<int>>& samples_data) {
		//assume that the size of all vector is equal.
		size_t n_samples = samples_data.size();
		size_t n_features = samples_data[0].size();
		sample_data_matrix.resize(n_samples, n_features);
		sample_data_cov.resize(n_features, n_features);
		for (size_t i = 0; i < n_samples; i++) {
			for (size_t j = 0; j < n_features; j++) {
				sample_data_matrix(i, j) = static_cast<double>(samples_data[i][j]);
			}
		}
	}

	void PCA::pca() {
		normalize();
		compute_cov();
		std::map<double, Eigen::VectorXd> result;
		double contribution;
		compute_lambda(result, contribution);
		
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sample_data_cov);
		Eigen::VectorXd all_eigenvalues = solver.eigenvalues();
		double total_variance = all_eigenvalues.sum();
	
		std::cout << "========== PCA Analysis Results ==========" << std::endl;
		std::cout << "Cumulative Contribution Rate: " << std::fixed << std::setprecision(4) << contribution * 100.0 << "%" << std::endl;
		std::cout << "Number of Principal Components: " << result.size() << std::endl;
		std::cout << std::endl;
		
		int component_idx = 1;
		double cumulative_contribution = 0.0;
		
		std::cout << "Principal Component Details (Sorted by Eigenvalue in Descending Order):" << std::endl;
		std::cout << std::string(80, '-') << std::endl;
		
		for (auto it = result.rbegin(); it != result.rend(); ++it) {
			double eigenvalue = it->first;
			const Eigen::VectorXd& eigenvector = it->second;
			
			double component_contribution = eigenvalue / total_variance;
			cumulative_contribution += component_contribution;
			
			std::cout << "Principal Component " << component_idx << ":" << std::endl;
			std::cout << "  Eigenvalue: " << std::fixed << std::setprecision(6) << eigenvalue << std::endl;
			std::cout << "  Contribution Rate: " << std::fixed << std::setprecision(4) << component_contribution * 100.0 << "%" << std::endl;
			std::cout << "  Cumulative Contribution Rate: " << std::fixed << std::setprecision(4) << cumulative_contribution * 100.0 << "%" << std::endl;
			std::cout << "  Eigenvector: [";
			for (Eigen::Index i = 0; i < eigenvector.size(); ++i) {
				std::cout << std::fixed << std::setprecision(4) << eigenvector(i);
				if (i < eigenvector.size() - 1) {
					std::cout << ", ";
				}
			}
			std::cout << "]" << std::endl;
			std::cout << std::endl;
			
			component_idx++;
		}
		
		std::cout << std::string(80, '=') << std::endl;
	}

	void PCA::compute_lambda(std::map<double, Eigen::VectorXd>& pca_lambda, double& contribution) {
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sample_data_cov);
		
		if (solver.info() != Eigen::Success) {
			contribution = 0.0;
			pca_lambda.clear();
			return;
		}
		
		Eigen::VectorXd eigenvalues = solver.eigenvalues();
		Eigen::MatrixXd eigenvectors = solver.eigenvectors();
		
		std::reverse(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
		eigenvectors = eigenvectors.rowwise().reverse().eval();
		
		double total_variance = eigenvalues.sum();
		
		if (total_variance < 1e-10) {
			contribution = 0.0;
			pca_lambda.clear();
			return;
		}
		
		double cumulative_variance = 0.0;
		int n_components = 0;
		
		for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
			cumulative_variance += eigenvalues(i);
			n_components++;
			contribution = cumulative_variance / total_variance;
			if (contribution >= 0.90) {
				break;
			}
		}
		pca_lambda.clear();
		for (int i = 0; i < n_components; ++i) {
			pca_lambda[eigenvalues(i)] = eigenvectors.col(i);
		}
	}

	void PCA::compute_cov() {
		Eigen::Index n_samples = sample_data_matrix.rows();
		Eigen::Index n_features = sample_data_matrix.cols();
		sample_data_cov.resize(n_features, n_features);
		sample_data_cov = (sample_data_matrix.transpose() * sample_data_matrix) / (n_samples - 1.0);
	}

	void PCA::normalize() {
		int col = sample_data_matrix.cols();
		for (int i = 0; i < col; i++) {
			double mean = 0.0;
			compute_mean(i, mean);
			double variance = 0.0;
			compute_variance(i, mean, variance);
			double std_dev = std::sqrt(variance);
			if (std_dev > 1e-10) {
				sample_data_matrix.col(i) = (sample_data_matrix.col(i).array() - mean) / std_dev;
			}
		}
	}

	inline void PCA::compute_mean(int& col, double& mean) {
		mean = sample_data_matrix.col(col).mean();
	}

	inline void PCA::compute_variance(int& col, double& mean, double& variance) {
		Eigen::VectorXd col_vec = sample_data_matrix.col(col);
		variance = (col_vec.array() - mean).square().mean();
	}
}
