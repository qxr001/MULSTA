#include "Cluster.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>
#include <iomanip>

namespace mulsta {
	Cluster::Cluster(std::vector<std::vector<double>>& samples_data) {
		if (samples_data.empty()) {
			sample_data_matrix = Eigen::MatrixXd::Zero(0, 0);
			return;
		}

		size_t rows = samples_data.size();
		size_t cols = samples_data[0].size();
		sample_data_matrix = Eigen::MatrixXd::Zero(rows, cols);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				sample_data_matrix(i, j) = samples_data[i][j];
			}
		}
	}

	void Cluster::shortest_distance() {
		if (sample_data_matrix.rows() == 0) {
			std::cout << "Error: Sample data is empty" << std::endl;
			return;
		}
		compute_shortest_distance();
	}

	void Cluster::shortest_distance_matrix(Eigen::MatrixXd& m) {
		size_t n = sample_data_matrix.rows();
		m = Eigen::MatrixXd::Zero(n, n);

		for (size_t i = 0; i < n; ++i) {
			for (size_t j = i + 1; j < n; ++j) {
				double dist = 0.0;
				for (int k = 0; k < sample_data_matrix.cols(); ++k) {
					double diff = sample_data_matrix(i, k) - sample_data_matrix(j, k);
					dist += diff * diff;
				}
				dist = std::sqrt(dist);
				m(i, j) = dist;
				m(j, i) = dist;
			}
		}
	}

	void Cluster::compute_shortest_distance() {
		size_t n = sample_data_matrix.rows();
		if (n == 0) return;

		Eigen::MatrixXd dist_matrix;
		shortest_distance_matrix(dist_matrix);

		std::vector<std::vector<size_t>> clusters;
		for (size_t i = 0; i < n; ++i) {
			clusters.push_back({ i });
		}

		std::cout << "Shortest distance clustering process:" << std::endl;
		std::cout << "Initial state: Each sample point is a cluster, total " << clusters.size() << " clusters" << std::endl;

		while (clusters.size() > 1) {
			double min_dist = std::numeric_limits<double>::max();
			size_t merge_i = 0, merge_j = 0;

			for (size_t i = 0; i < clusters.size(); ++i) {
				for (size_t j = i + 1; j < clusters.size(); ++j) {
					double cluster_dist = std::numeric_limits<double>::max();
					for (size_t idx_i : clusters[i]) {
						for (size_t idx_j : clusters[j]) {
							if (dist_matrix(idx_i, idx_j) < cluster_dist) {
								cluster_dist = dist_matrix(idx_i, idx_j);
							}
						}
					}

					if (cluster_dist < min_dist) {
						min_dist = cluster_dist;
						merge_i = i;
						merge_j = j;
					}
				}
			}

			std::cout << "Merge cluster " << merge_i << " and cluster " << merge_j 
				<< ", distance: " << min_dist << ", remaining clusters: " << clusters.size() - 1 << std::endl;

			clusters[merge_i].insert(clusters[merge_i].end(), 
				clusters[merge_j].begin(), clusters[merge_j].end());

			clusters.erase(clusters.begin() + merge_j);

			std::cout << "Current clustering result:" << std::endl;
			for (size_t c = 0; c < clusters.size(); ++c) {
				std::cout << "Cluster " << c << ": {";
				for (size_t idx = 0; idx < clusters[c].size(); ++idx) {
					std::cout << clusters[c][idx];
					if (idx < clusters[c].size() - 1) std::cout << ", ";
				}
				std::cout << "}" << std::endl;
			}
			std::cout << std::endl;
		}

		std::cout << "Clustering completed! All sample points merged into one cluster." << std::endl;
	}

	void Cluster::kmeans(int k, int max_iterations) {
		if (sample_data_matrix.rows() == 0) {
			std::cout << "Error: Sample data is empty" << std::endl;
			return;
		}

		if (k <= 0 || k > sample_data_matrix.rows()) {
			std::cout << "Error: Invalid number of clusters k. Must be between 1 and " 
				<< sample_data_matrix.rows() << std::endl;
			return;
		}

		std::cout << "K-means clustering with k = " << k << std::endl;
		std::cout << "Number of samples: " << sample_data_matrix.rows() 
			<< ", Number of features: " << sample_data_matrix.cols() << std::endl;

		// Initialize cluster centroids
		initialize_centroids(k);

		// Initialize cluster labels
		cluster_labels.resize(sample_data_matrix.rows(), -1);

		// Iteratively execute k-means
		for (int iter = 0; iter < max_iterations; ++iter) {
			Eigen::MatrixXd old_centroids = centroids;

			// Assign each point to the nearest cluster centroid
			assign_clusters();

			// Update cluster centroids
			update_centroids();

			// Check for convergence
			if (check_convergence(old_centroids)) {
				std::cout << "Converged after " << (iter + 1) << " iterations." << std::endl;
				break;
			}

			if (iter == max_iterations - 1) {
				std::cout << "Reached maximum iterations (" << max_iterations << ")." << std::endl;
			}
		}

		// Output clustering results
		std::cout << "\nK-means clustering results:" << std::endl;
		for (int c = 0; c < k; ++c) {
			std::vector<int> cluster_points;
			for (size_t i = 0; i < cluster_labels.size(); ++i) {
				if (cluster_labels[i] == c) {
					cluster_points.push_back(i);
				}
			}
			std::cout << "Cluster " << c << " (" << cluster_points.size() << " points): {";
			for (size_t idx = 0; idx < cluster_points.size(); ++idx) {
				std::cout << cluster_points[idx];
				if (idx < cluster_points.size() - 1) std::cout << ", ";
			}
			std::cout << "}" << std::endl;
			std::cout << "  Centroid: [";
			for (int j = 0; j < centroids.cols(); ++j) {
				std::cout << std::fixed << std::setprecision(4) << centroids(c, j);
				if (j < centroids.cols() - 1) std::cout << ", ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << std::endl;
	}

	void Cluster::initialize_centroids(int k) {
		int n_samples = sample_data_matrix.rows();
		int n_features = sample_data_matrix.cols();
		centroids = Eigen::MatrixXd::Zero(k, n_features);

		// Use random initialization: randomly select k distinct sample points as initial cluster centroids
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, n_samples - 1);

		std::vector<int> selected_indices;
		for (int i = 0; i < k; ++i) {
			int idx;
			do {
				idx = dis(gen);
			} while (std::find(selected_indices.begin(), selected_indices.end(), idx) != selected_indices.end());
			
			selected_indices.push_back(idx);
			for (int j = 0; j < n_features; ++j) {
				centroids(i, j) = sample_data_matrix(idx, j);
			}
		}

		std::cout << "Initialized " << k << " centroids randomly." << std::endl;
	}

	double Cluster::euclidean_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
		return (a - b).norm();
	}

	void Cluster::assign_clusters() {
		int n_samples = sample_data_matrix.rows();
		int k = centroids.rows();

		for (int i = 0; i < n_samples; ++i) {
			double min_dist = std::numeric_limits<double>::max();
			int closest_cluster = 0;

			Eigen::VectorXd sample = sample_data_matrix.row(i);

			for (int c = 0; c < k; ++c) {
				Eigen::VectorXd centroid = centroids.row(c);
				double dist = euclidean_distance(sample, centroid);

				if (dist < min_dist) {
					min_dist = dist;
					closest_cluster = c;
				}
			}

			cluster_labels[i] = closest_cluster;
		}
	}

	void Cluster::update_centroids() {
		int n_samples = sample_data_matrix.rows();
		int n_features = sample_data_matrix.cols();
		int k = centroids.rows();

		Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, n_features);
		std::vector<int> cluster_counts(k, 0);

		// Calculate the sum of all points in each cluster
		for (int i = 0; i < n_samples; ++i) {
			int cluster = cluster_labels[i];
			cluster_counts[cluster]++;
			for (int j = 0; j < n_features; ++j) {
				new_centroids(cluster, j) += sample_data_matrix(i, j);
			}
		}

		// Calculate the mean (new cluster centroids)
		for (int c = 0; c < k; ++c) {
			if (cluster_counts[c] > 0) {
				for (int j = 0; j < n_features; ++j) {
					new_centroids(c, j) /= cluster_counts[c];
				}
			}
		}

		centroids = new_centroids;
	}

	bool Cluster::check_convergence(const Eigen::MatrixXd& old_centroids, double tolerance) {
		if (centroids.rows() != old_centroids.rows() || centroids.cols() != old_centroids.cols()) {
			return false;
		}

		for (int i = 0; i < centroids.rows(); ++i) {
			for (int j = 0; j < centroids.cols(); ++j) {
				if (std::abs(centroids(i, j) - old_centroids(i, j)) > tolerance) {
					return false;
				}
			}
		}

		return true;
	}
}