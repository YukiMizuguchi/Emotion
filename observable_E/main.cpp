#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <array>
#include <numeric>
#include <iterator>
#include <tuple>
#include <cmath>
#include <list>
#include <set>

#define F "result.txt"

using namespace std;
const int typeN = 216; //3*3*3*2*2*2


const double mutationRate = 0.025;
const double shockRate = 0.1;
const double b = 1.0; //協力による利益
const double c = 0.3; //協力コスト
const double c_e = 0.1; //感情表現コスト
const int generationN = 10000; // 3000
const int learningP = 200; //3000
const int d = 2;
const double delta = 0.25;


void Learning(vector<vector<double>>& x, vector<vector<double>>& x_new, double* mu, vector<double>&mu_e_t);
double calculateUtility(double x1, double x2, int id1, int id2);
void calculateFitness(double* mu, vector<vector<double>>& x, double* ave_payoff_list);
void calculateFrequency(double* new_mu, double* mu, double* ave_payoff_list);
void mutation(vector<vector<double>>& x_t, double* mu_new, double rnd);
void randomShock(vector<vector<double>>& x, double* mu);
void outputResult(int generation, double* mu, vector<double>& mu_e, double* ave_payoff_list, vector<vector<double>>& x);
vector<double> calculateMuE(double* mu_t);
vector<vector<double>> copyArray(const vector<std::vector<double>>& x_new);

ofstream outputfile(F);

int main()
{
	srand((unsigned)time(NULL)); // 乱数シード初期化
	std::mt19937_64 mt64(rand());  //もとにする整数型に乱数を用いる
	std::uniform_real_distribution<double> uni(0, 1);
	vector<vector<double>> x(typeN, vector<double>(27, 0.5));
	vector<vector<double>> x_new(typeN, vector<double>(27));

	double mu[typeN];
	double mu_new[typeN];
	double ave_payoff_list[typeN] = {0.0};

	vector<double> mu_e(27);


    for(int i = 0; i < typeN; ++i) {
        mu[i] = 1.0/typeN;
    }

    for (int i = 0; i < generationN; i++) {
    	cout << "generation: "<< i << endl;
    	mu_e = calculateMuE(mu);
        for (int j = 0; j < learningP; j++) {
            Learning(x, x_new, mu, mu_e); //学習
            x = copyArray(x_new);
        }

        calculateFitness(mu, x, ave_payoff_list); //選ばれた行動から利得を計算

        outputResult(i, mu, mu_e, ave_payoff_list, x);

        calculateFrequency(mu_new, mu, ave_payoff_list); //次世代の割合を計算

        if (uni(mt64) < mutationRate){
            mutation(x, mu_new, uni(mt64));
            cout << "mutation" << endl;
        }
        copy(mu_new, mu_new + typeN, mu);

        if (uni(mt64) < shockRate){
        	cout << "random shock" << endl;
        	randomShock(x, mu);
        }

    }

    outputfile.close();

    cout << "Simulation complete." << endl;
}

vector<vector<double>> copyArray(const vector<std::vector<double>>& x_new_t) {
    vector<vector<double>> x_t = x_new_t;  // 単純な代入でコピー可能
    return x_t;
}

void Learning(vector<vector<double>>& x_t, vector<vector<double>>& x_new_t, double* mu_t, vector<double>& mu_e_t){ //配列の返し方
	vector<vector<double>> x_e(27, vector<double>(27, 0.0));
	for (int k = 0; k < typeN; k++){
		int e_id = k%27;
		if (mu_e_t[e_id] > 0.00001){
			for (int l = 0; l < 27; l++){
				x_e[e_id][l] += x_t[k][l]*mu_t[k]/mu_e_t[e_id];
			}
		}
		//mu_e_t[e_id] < 0.00001の場合、mu_t[k] < 0.00001となるので学習されない
	}

    for (int k = 0; k < typeN; k++){
        if (mu_t[k] > 0.00001){ //集団中に存在するタイプのみ学習
        	int e_id1 = k%27;
            for (int e_id2 = 0; e_id2 < 27; e_id2++){
                x_new_t[k][e_id2] = x_t[k][e_id2]+mu_e_t[e_id2]*delta*x_t[k][e_id2]*(calculateUtility(1.0,x_e[e_id2][e_id1],k,e_id2)-calculateUtility(x_t[k][e_id2],x_e[e_id2][e_id1],k,e_id2));
            }
        }
    }
}

vector<double> calculateMuE(double* mu_t){
	vector<double> mu_e_t(27, 0.0);
	for (int k = 0; k < typeN; k++){
		mu_e_t[k%27] += mu_t[k];
	}
	return mu_e_t;
}

double calculateUtility(double x1, double x2, int id1, int id2){
    int r1_1 = (id1-(id1%108))/108;
    int r2_1 = ((id1%108)-(id1%108)%54)/54;
    int r3_1 = ((id1%54)-(id1%54)%27)/27;
    int e1_1 = ((id1%27)-(id1%27)%9)/9-1;
    int e2_1 = ((id1%9)-(id1%9)%3)/3-1;
    int e3_1 = id1%3-1;

    int e1_2 = ((id2%27)-(id2%27)%9)/9-1;
    int e2_2 = ((id2%9)-(id2%9)%3)/3-1;
    int e3_2 = id2%3-1;

    double u = x1*x2*(b-c-c_e*abs(e1_1)+r1_1*e1_2)+x1*(1-x2)*(-c-c_e*abs(e3_1)+r3_1*e2_2)+(1-x1)*x2*(b-c_e*abs(e2_1)+r2_1*e3_2)+(1-x1)*(1-x2)*0;

    return u;
}

void calculateFitness(double* mu_t, vector<vector<double>>& x_t, double* ave_payoff_list_t){
    for (int k = 0; k < typeN; k++){
        if (mu_t[k] > 0.0000001){ //集団中に存在するタイプのみ計算
            int e1_1 = ((k%27)-(k%27)%9)/9-1;
            int e2_1 = ((k%9)-(k%9)%3)/3-1;
            int e3_1 = k%3-1;
            int e_id1 = k%27;

            for (int l = 0; l < typeN; l++){
            	int e_id2 = l%27;
                ave_payoff_list_t[k] += mu_t[l]*(x_t[k][e_id2]*x_t[l][e_id1]*(b+c_e*(1-abs(e1_1)))+x_t[k][e_id2]*(1-x_t[l][e_id1])*(c_e*(1-abs(e3_1)))+(1-x_t[k][e_id2])*x_t[l][e_id1]*(b+c+c_e*(1-abs(e2_1)))+(1-x_t[k][e_id2])*(1-x_t[l][e_id1])*(c+c_e));
            }
        }
    }
}

void calculateFrequency(double* new_mu_t, double* mu_t, double* ave_payoff_list_t){
    double ave_ave_fit = 0;
    for (int j = 0; j < typeN; j++) {
        ave_ave_fit += mu_t[j]*ave_payoff_list_t[j];
    }

    for (int j = 0; j < typeN; j++) {
        new_mu_t[j] = mu_t[j]*(d + ave_payoff_list_t[j])/(d + ave_ave_fit);
        ave_payoff_list_t[j] = 0; //利得のリセット
    }
}

void mutation(vector<vector<double>>& x_t, double* mu_new_t, double rnd){
    int id = rand()%216;
    if(mu_new_t[id] <= 0.0000001){
    	for (int l = 0; l < 8; l++){
    		x_t[id][l] = 0.5;
    	}
    }
	//int id = 127;
    mu_new_t[id] = mu_new_t[id] + (rnd/10.0);
    for (int j = 0; j < typeN; j++) {
        mu_new_t[j] = mu_new_t[j]/(1+rnd/10.0);
    }
}


void randomShock(vector<vector<double>>& x, double* mu){
	std::mt19937_64 mt64_t(rand());  //もとにする整数型に乱数を用いる
	int id = rand()%216;
	while (mu[id] < 0.00001){
		id = rand()%216;
	}
	for (int k = 0; k < 27; k++) {
		std::normal_distribution<double> normal(x[id][k], 0.2);
		double number = 2.0;
		while ((number >= 1.0) or (number <= 0.0)){
			number = normal(mt64_t);
		}
		x[id][k] = number;
	}
}


void outputResult(int generation, double* mu_t, vector<double>& mu_e_t, double* ave_payoff_list_t, vector<vector<double>>& x_t){
	double r1;
	double r2;
    double r3;
    int e1;
	int e2;
    int e3;
    double CRate[typeN] = {0.0};
    double TotalCRate = 0.0;
	for (int j = 0; j < typeN; j++) {
		r1 = (j-(j%108))/108;
		r2 = ((j%108)-(j%108)%54)/54;
		r3 = ((j%54)-(j%54)%27)/27;
		e1 = ((j%27)-(j%27)%9)/9-1;
		e2 = ((j%9)-(j%9)%3)/3-1;
		e3 = j%3-1;

        for (int k = 0; k < 27; k++) {
        	CRate[j] += mu_e_t[k]*x_t[j][k];
        }
        TotalCRate += mu_t[j]*CRate[j];
		outputfile<< generation << "\t" << j << "\t" << r1 << "\t" << r2 << "\t" << r3 << "\t" << e1 << "\t" << e2 << "\t" << e3 << "\t"<< mu_t[j]<< "\t" << ave_payoff_list_t[j] << "\t" << CRate[j] << "\t" << TotalCRate;
		outputfile<<"\n";
	}
}
