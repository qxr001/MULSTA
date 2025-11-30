#pragma once
#include <vector>
#include <map>
#include <Eigen/Dense>
namespace mulsta {
	class PCA {
	public:
		PCA(std::vector<int>& adaMath,
			std::vector<int>& proSta,
			std::vector<int>& linAlge,
			std::vector<int>& colEng,
			std::vector<int>& statt,
			std::vector<int>& python);
		PCA(std::vector<std::vector<int>>& samples_data);
		~PCA() {}
	public:
		void pca();

	protected:
		void compute_cov();
		void compute_lambda(std::map<double, Eigen::VectorXd>& pca_lambda, double& contribution);

		void normalize();

		inline void compute_mean(int& col, double& mean);
		inline void compute_variance(int& col, double& mean, double& variance);
	protected:
		Eigen::MatrixXd sample_data_matrix;
		Eigen::MatrixXd sample_data_cov;
	};
	
}
