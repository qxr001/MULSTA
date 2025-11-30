#pragma once
#include <vector>
#include <Eigen/Eigen>

namespace mulsta {
	//Discriminant Analysis
	class DisAna {
	public:
		DisAna(std::vector<std::vector<std::vector<double>>>& samples_data,
			std::vector<double>& data1, std::vector<double>& data2);
		~DisAna() {}

	public:
		void dis_Ana();

	protected:
		void normlize();
//		void compute_mean(std::vector<double>& col, double& mean);
		void compute_mean(int& i, int& j, double& mean);
//		void compute_varience(std::vector<double>& col, double& mean, double& varience);
		void compute_varience(int& i, int& j, double& mean, double& varience);

		void compute_cov();

	protected:
		std::vector<Eigen::MatrixXd> sample_data_matrixs;
		std::vector<Eigen::MatrixXd> sample_data_covs;
		std::vector<std::vector<double>> unchecked_data;
		std::vector<std::vector<double>> original_means;
		std::vector<std::vector<double>> original_stds;
	};
}
