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
const double b = 1.0; //協力による利益
const double c = 0.3; //協力コスト
const double c_e = 0.1; //感情表現コスト
const int generationN = 350; //300世代程度で収束
const int d = 2;

void calculateNash(std::vector<std::vector<std::vector<int>>>& eq_vec, std::vector<std::vector<std::vector<double>>>& mix_vec, int id1, int id2);
void calculateFitness(double* mu, std::vector<std::vector<std::vector<int>>>& eq_vec, std::vector<std::vector<std::vector<double>>>& mix_vec, vector<vector<double>>& x, double* ave_payoff_list);
void calculateFrequency(double* new_mu, double* mu, double* ave_payoff_list);
void mutation(double* mu_new, double rnd);
void outputResult(int generation, double* mu, double* ave_payoff_list, vector<vector<double>>& x);


ofstream outputfile(F);

int main()
{
	srand((unsigned)time(NULL)); // 乱数シード初期化
	//0から1の乱数を生成
	std::mt19937_64 mt64(rand());  //もとにする整数型に乱数を用いる
	std::uniform_real_distribution<double> uni(0, 1);

	double mu[typeN];
	double mu_new[typeN];
	double ave_payoff_list[typeN] = {0.0};

    for(int i = 0; i < typeN; ++i) {
        mu[i] = 1.0/typeN;
    }
    std::vector<std::vector<std::vector<int>>> eq_vec;
    std::vector<std::vector<std::vector<double>>> mix_vec;
    for (int j = 0; j < typeN; ++j) { // 3レイヤーを追加する例
    	eq_vec.push_back({}); // 新しいレイヤーを追加

        // 各レイヤーに行を追加していく
        for (int k = 0; k <= typeN; ++k) { // 行数を異なる数にする
        	eq_vec[j].push_back({}); // 新しい行を追加
        }
    }

    for (int j = 0; j < typeN; ++j) { // 3レイヤーを追加する例
     	mix_vec.push_back({}); // 新しいレイヤーを追加

    	// 各レイヤーに行を追加していく
    	for (int k = 0; k <= typeN; ++k) { // 行数を異なる数にする
    		mix_vec[j].push_back({}); // 新しい行を追加
    	}
    }

    for (int id1 = 0; id1 < typeN; ++id1) {
    	for (int l = 0; l < typeN; ++l){
        	calculateNash(eq_vec, mix_vec, id1, l);
        }
    }

    for (int i = 0; i < generationN; i++) {
    	cout << "generation: "<< i << endl;
    	vector<vector<double>> x(typeN, vector<double>(typeN, 0.0));
        calculateFitness(mu, eq_vec, mix_vec, x, ave_payoff_list); //選ばれた行動から利得を計算

        outputResult(i, mu, ave_payoff_list, x);

        calculateFrequency(mu_new, mu, ave_payoff_list); //次世代の割合を計算

        if (uni(mt64) < mutationRate){
            mutation(mu_new, uni(mt64));
            cout << "mutation" << endl;
        }
        copy(mu_new, mu_new + typeN, mu);
    }

    outputfile.close();

    cout << "Simulation complete." << endl;
}

void calculateNash(std::vector<std::vector<std::vector<int>>>& eq_vec, std::vector<std::vector<std::vector<double>>>& mix_vec, int id1, int id2){
    int r1_1 = (id1-(id1%108))/108;
    int r2_1 = ((id1%108)-(id1%108)%54)/54;
    int r3_1 = ((id1%54)-(id1%54)%27)/27;
    int e1_1 = ((id1%27)-(id1%27)%9)/9-1;
    int e2_1 = ((id1%9)-(id1%9)%3)/3-1;
    int e3_1 = id1%3-1;
    int r1_2 = (id2-(id2%108))/108;
    int r2_2 = ((id2%108)-(id2%108)%54)/54;
    int r3_2 = ((id2%54)-(id2%54)%27)/27;
    int e1_2 = ((id2%27)-(id2%27)%9)/9-1;
    int e2_2 = ((id2%9)-(id2%9)%3)/3-1;
    int e3_2 = id2%3-1;

    double CC_P1 = b - c + r1_1*e1_2 - c_e*abs(e1_1);
    double CD_P1 = -c + r3_1*e2_2 - c_e*abs(e3_1);
    double DC_P1 = b + r2_1*e3_2 - c_e*abs(e2_1);
    double DD_P1 = 0;
    double CC_P2 = b - c + r1_2*e1_1 - c_e*abs(e1_2);
    double CD_P2 = b + r2_2*e3_1 - c_e*abs(e2_2);
    double DC_P2 = -c + r3_2*e2_1 - c_e*abs(e3_2);
    double DD_P2 = 0;
    double Mix_P1;
    double Mix_P2;

    double x1_mix = -1.0;
    double x2_mix = -1.0;


    if (abs(CC_P1-CD_P1-DC_P1+DD_P1) > 0.001 && abs(CC_P2-CD_P2-DC_P2+DD_P2) > 0.001){
    	x1_mix = (-DC_P2+DD_P2)/(CC_P2-CD_P2-DC_P2+DD_P2);
        x2_mix = (-CD_P1+DD_P1)/(CC_P1-CD_P1-DC_P1+DD_P1);
    }
    if (x1_mix > 0.0 && x1_mix < 1.0 && x2_mix > 0.0 && x2_mix < 1.0){
        Mix_P1 = x1_mix*x2_mix*CC_P1 + x1_mix*(1-x2_mix)*CD_P1 + (1-x1_mix)*x2_mix*DC_P1 + (1-x1_mix)*(1-x2_mix)*DD_P1 + 10;
        Mix_P2 = x1_mix*x2_mix*CC_P2 + x1_mix*(1-x2_mix)*CD_P2 + (1-x1_mix)*x2_mix*DC_P2 + (1-x1_mix)*(1-x2_mix)*DD_P2 + 10;
    }else{
        Mix_P1 = -1.0;
        Mix_P2 = -1.0;
    }

    double nash_list[2][5] = {{CC_P1, CD_P1, DC_P1, DD_P1, Mix_P1}, {CC_P2, CD_P2, DC_P2, DD_P2, Mix_P2}};
    if ((CC_P1 >= DC_P1) && (CC_P2 >= CD_P2)){
        nash_list[0][0] += 10;
        nash_list[1][0] += 10;
    }
    if ((CD_P1 >= DD_P1) && (CD_P2 >= CC_P2)){
        nash_list[0][1] += 10;
        nash_list[1][1] += 10;
    }
    if ((DC_P1 >= CC_P1) && (DC_P2 >= DD_P2)){
        nash_list[0][2] += 10;
        nash_list[1][2] += 10;
    }
    if ((DD_P1 >= CD_P1) && (DD_P2 >= DC_P2)){
        nash_list[0][3] += 10;
        nash_list[1][3] += 10;
    }

    for (int j = 0; j < 5; j++) {
    	bool efficient = true;
        for (int k = 0; k < 5; k++) {
        	if ((nash_list[0][k] > nash_list[0][j]) && (nash_list[1][k] > nash_list[1][j])){
        		efficient = false;
            }
        }
        if (efficient){
        	eq_vec[id1][id2].push_back(j);
        	if (j == 4){
        		mix_vec[id1][id2].push_back(Mix_P1-10);
        		mix_vec[id1][id2].push_back(Mix_P2-10);
        	}
        }
    }
}

void calculateFitness(double* mu_t, std::vector<std::vector<std::vector<int>>>& eq_vec, std::vector<std::vector<std::vector<double>>>& mix_vec, vector<vector<double>>& x, double* ave_payoff_list_t){
    for (int k = 0; k < typeN; k++){
        if (mu_t[k] > 0.0000001){ //集団中に存在するタイプのみ計算
            int e1_1 = abs(((k%27)-(k%27)%9)/9-1);
            int e2_1 = abs(((k%9)-(k%9)%3)/3-1);
            int e3_1 = abs(k%3-1);

            for (int l = 0; l < typeN; l++){
            	int size = eq_vec[k][l].size();
            	for (int a = 0; a < size; a++){
            		vector<vector<double>> x_t(typeN, vector<double>(typeN, 0));
            		int eq = eq_vec[k][l][a];
					if (eq == 0){
						x_t[k][l] = 1.0;
						x_t[l][k] = 1.0;
					}else if (eq == 1){
						x_t[k][l] = 1.0;
						x_t[l][k] = 0.0;
					}else if (eq == 2){
						x_t[k][l] = 0.0;
						x_t[l][k] = 1.0;
					}else if (eq == 3){
						x_t[k][l] = 0.0;
						x_t[l][k] = 0.0;
					}else if (eq == 4){
						x_t[k][l] = mix_vec[k][l][0];
						x_t[l][k] = mix_vec[k][l][1];
					}else {
						cout << "Error2!!" << endl;
					}


					ave_payoff_list_t[k] += mu_t[l]*(x_t[k][l]*x_t[l][k]*(b+c_e*(1-e1_1))+x_t[k][l]*(1-x_t[l][k])*(c_e*(1-e3_1))+(1-x_t[k][l])*x_t[l][k]*(b+c+c_e*(1-e2_1))+(1-x_t[k][l])*(1-x_t[l][k])*(c+c_e))/size;

					if (eq == 0){
						x[k][l] += 1.0/size;
					}else if (eq == 1){
						x[k][l] += 1.0/size;
					}else if (eq == 2){
						x[k][l] += 0.0;
					}else if (eq == 3){
						x[k][l] += 0.0;
					}else if (eq == 4){
						x[k][l] += mix_vec[k][l][0]/size;
					}else {
						cout << "Error2!!" << endl;
					}
            	}
            }
        }
    }
}

void calculateFrequency(double* new_mu_t, double* mu_t, double* ave_payoff_list_t){
    double ave_ave_fit = 0.0;
    for (int j = 0; j < typeN; j++) {
        ave_ave_fit += mu_t[j]*ave_payoff_list_t[j];
    }

    for (int j = 0; j < typeN; j++) {
        new_mu_t[j] = mu_t[j]*(d + ave_payoff_list_t[j])/(d + ave_ave_fit);
        ave_payoff_list_t[j] = 0; //利得のリセット
    }
}

void mutation(double* mu_new_t, double rnd){// 0.025
    int id = rand()%216;
    mu_new_t[id] = mu_new_t[id] + (rnd/10.0);
    for (int j = 0; j < typeN; j++) {
        mu_new_t[j] = mu_new_t[j]/(1+(rnd/10.0));
    }
}


void outputResult(int generation, double* mu_t, double* ave_payoff_list_t, vector<vector<double>>& x_t){
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

        for (int k = 0; k < typeN; k++) {
        	CRate[j] += mu_t[k]*x_t[j][k];
        }
        TotalCRate += mu_t[j]*CRate[j];
		outputfile<< generation << "\t" << j << "\t" << r1 << "\t" << r2 << "\t" << r3 << "\t" << e1 << "\t" << e2 << "\t" << e3 << "\t"<< mu_t[j]<< "\t" << ave_payoff_list_t[j] << "\t" << CRate[j] << "\t" << TotalCRate;
		outputfile<<"\n";
	}
}
