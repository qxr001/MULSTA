#include "DisAna.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace mulsta {
	DisAna::DisAna(std::vector<std::vector<std::vector<double>>>& samples_data,
		std::vector<double>& data1, std::vector<double>& data2) {
		//1. init the unchecked data
		unchecked_data.clear();
		unchecked_data.push_back(data1);
		unchecked_data.push_back(data2);

		//2. init the matrix
		int n = samples_data.size();
		sample_data_matrixs.resize(n);
		sample_data_covs.resize(n);
		// init for every sample_data_matrix
		for (size_t i = 0; i < n; i++) {
			if (samples_data[i].empty()) {
				sample_data_matrixs[i] = Eigen::MatrixXd::Zero(0, 0);
				continue;
			}
			size_t rows = samples_data[i].size();
			size_t cols = samples_data[i][0].size();
			sample_data_matrixs[i] = Eigen::MatrixXd::Zero(rows, cols);
			for (size_t j = 0; j < rows; j++) {
				for (size_t k = 0; k < cols; k++) {
					sample_data_matrixs[i](j, k) = samples_data[i][j][k];
				}
			}
		}
	}

	void DisAna::dis_Ana() {
		int n_classes = sample_data_matrixs.size();
		original_means.clear();
		original_stds.clear();
		original_means.resize(n_classes);
		original_stds.resize(n_classes);
		
		for (int i = 0; i < n_classes; i++) {
			int n_features = sample_data_matrixs[i].cols();
			original_means[i].resize(n_features);
			original_stds[i].resize(n_features);
			
			for (int j = 0; j < n_features; j++) {
				double mean = 0.0;
				compute_mean(i, j, mean);
				original_means[i][j] = mean;
				double variance = 0.0;
				compute_varience(i, j, mean, variance);
				double std_dev = std::sqrt(variance);
				original_stds[i][j] = std_dev;
			}
		}
		
		normlize();
		compute_cov();
		
		std::cout << "========== Discriminant Analysis Results ==========" << std::endl;
		for (size_t idx = 0; idx < unchecked_data.size(); idx++) {
			int n_features = unchecked_data[idx].size();
			Eigen::VectorXd x = Eigen::VectorXd::Zero(n_features);
			for (int i = 0; i < n_features; i++) {
				x(i) = unchecked_data[idx][i];
			}
			
			std::vector<double> mahalanobis_distances(n_classes);
			for (int i = 0; i < n_classes; i++) {
				Eigen::VectorXd x_normalized = x;
				Eigen::VectorXd mean_vec = Eigen::VectorXd::Zero(n_features);
				
				for (int j = 0; j < n_features; j++) {
					double mean = original_means[i][j];
					double std_dev = original_stds[i][j];
					mean_vec(j) = 0.0;
					
					if (std_dev > 1e-10) {
						x_normalized(j) = (x(j) - mean) / std_dev;
					} else {
						x_normalized(j) = x(j) - mean;
					}
				}
				
				Eigen::VectorXd diff = x_normalized - mean_vec;

				Eigen::MatrixXd cov_inv = sample_data_covs[i].inverse();
				
				double mahalanobis_dist_sq = diff.transpose() * cov_inv * diff;
				mahalanobis_distances[i] = std::sqrt(mahalanobis_dist_sq);
			}
			
			int min_class = 0;
			double min_distance = mahalanobis_distances[0];
			for (int i = 1; i < n_classes; i++) {
				if (mahalanobis_distances[i] < min_distance) {
					min_distance = mahalanobis_distances[i];
					min_class = i;
				}
			}
			
			std::cout << "Data point " << (idx + 1) << ":" << std::endl;
			std::cout << "  Input: [";
			for (size_t i = 0; i < unchecked_data[idx].size(); i++) {
				std::cout << std::fixed << std::setprecision(2) << unchecked_data[idx][i];
				if (i < unchecked_data[idx].size() - 1) std::cout << ", ";
			}
			std::cout << "]" << std::endl;
			std::cout << "  Mahalanobis distances to each class:" << std::endl;
			for (int i = 0; i < n_classes; i++) {
				std::cout << "    Class " << (i + 1) << ": " << std::fixed << std::setprecision(4) 
					<< mahalanobis_distances[i] << std::endl;
			}
			std::cout << "  Classification result: Class " << (min_class + 1) 
				<< " (minimum distance: " << std::fixed << std::setprecision(4) 
				<< min_distance << ")" << std::endl;
			std::cout << std::endl;
		}
		std::cout << std::string(80, '=') << std::endl;
	}

	void DisAna::compute_cov() {
		int n = sample_data_matrixs.size();
		for (int i = 0; i < n; i++) {
			Eigen::Index n_samples = sample_data_matrixs[i].rows();
			Eigen::Index n_features = sample_data_matrixs[i].cols();
			sample_data_covs[i].resize(n_features, n_features);
			sample_data_covs[i] = (sample_data_matrixs[i].transpose() * sample_data_matrixs[i]) / (n_samples - 1.0);
		}
	}


	void DisAna::normlize() {
		int n = sample_data_matrixs.size();
		for (int i = 0; i < n; i++) {
			int cols = sample_data_matrixs[i].cols();
			for (int j = 0; j < cols; j++) {
				double mean = 0.0;
				compute_mean(i, j, mean);
				double variance = 0.0;
				compute_varience(i, j, mean, variance);
				double std_dev = std::sqrt(variance);
				if (std_dev > 1e-10) {
					sample_data_matrixs[i].col(j) = (sample_data_matrixs[i].col(j).array() - mean) / std_dev;
				}
			}
		}
	}
	void DisAna::compute_mean(int& i, int& j, double& mean) {
		mean = sample_data_matrixs[i].col(j).mean();
	}
	void DisAna::compute_varience(int& i, int& j, double& mean, double& varience) {
		Eigen::VectorXd col_vec = sample_data_matrixs[i].col(j);
		varience = (col_vec.array() - mean).square().mean();
	}

}
