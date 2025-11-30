#pragma once
#include <vector>
#include <map>
#include <Eigen/Eigen>
namespace mulsta {
	class Cluster {
	public:
		Cluster(std::vector<std::vector<double>>& samples_data);
		~Cluster() {}
	public:
		void shortest_distance();
		void kmeans(int k, int max_iterations = 100);
		std::vector<int> get_cluster_labels() const { return cluster_labels; }
		Eigen::MatrixXd get_centroids() const { return centroids; }
	protected:
		void shortest_distance_matrix(Eigen::MatrixXd& m);
		void compute_shortest_distance();
		void initialize_centroids(int k);
		double euclidean_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
		void assign_clusters();
		void update_centroids();
		bool check_convergence(const Eigen::MatrixXd& old_centroids, double tolerance = 1e-6);
	protected:
		Eigen::MatrixXd sample_data_matrix;
		Eigen::MatrixXd centroids;
		std::vector<int> cluster_labels;
	};
}
