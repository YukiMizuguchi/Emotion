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

const int typeN = 216; //タイプ数（3*3*3*2*2*2）
const double mutationRate = 0.025;
const double shockRate = 0.1;
const double b = 1.0; //協力による利益
const double c = 0.3; //協力コスト
const double c_e = 0.1; //感情表現コスト
const int generationN = 300; // 世代数
const int learningP = 200; //学習ステップ数
const int d = 2; //進化速度定数
const double delta = 0.25; //学習速度定数


void Learning(vector<vector<double>>& x, vector<vector<double>>& x_new, double* mu, vector<double>&mu_r_t);
double calculateUtility(double x1, int id1, int r_id2, vector<vector<double>>& x_t, double* mu_t, vector<double>& mu_r_t);
void calculateFitness(double* mu, vector<vector<double>>& x, double* ave_payoff_list);
void calculateFrequency(double* new_mu, double* mu, double* ave_payoff_list);
void mutation(vector<vector<double>>& x, double* mu_new, double rnd);
void randomShock(vector<vector<double>>& x, double* mu);
void outputResult(int generation, double* mu, vector<double>& mu_r, double* ave_payoff_list, vector<vector<double>>& x);
vector<double> calculateMuR(double* mu_t);
vector<vector<double>> copyArray(const vector<std::vector<double>>& x_new);

ofstream outputfile(F);

int main()
{
	srand((unsigned)time(NULL)); // 乱数シード初期化
	std::mt19937_64 mt64(rand());  //もとにする整数型に乱数を用いる
	std::uniform_real_distribution<double> uni(0, 1); //0から1の乱数を生成。

	vector<vector<double>> x(typeN, vector<double>(8, 0.5)); // 戦略（協力率）のベクトル（初期値は0.5）
	vector<vector<double>> x_new(typeN, vector<double>(8)); // 戦略（協力率）のベクトル（コピー用）
	vector<double> mu_r(8); // 各rの割合の配列
	double mu[typeN]; // 各タイプの割合の配列
	double mu_new[typeN]; // 各タイプの割合の配列（コピー用）
	double ave_payoff_list[typeN] = {0.0}; // 各タイプが得た適応度の配列

    // 割合の初期化（均等に配分）
    for(int i = 0; i < typeN; ++i) {
        mu[i] = 1.0/typeN;
    }

    // シミュレーション開始
    for (int i = 0; i < generationN; i++) {
    	cout << "generation: "<< i << endl;
    	mu_r = calculateMuR(mu); // 各rのタイプ割合の取得

        // 戦略（各rに対する協力率）の学習
        for (int j = 0; j < learningP; j++) {
            Learning(x, x_new, mu, mu_r); // 学習
            x = copyArray(x_new); // ベクトルのコピー
        }
        calculateFitness(mu, x, ave_payoff_list); // 学習した協力率を用いて平均適応度を計算

        outputResult(i, mu, mu_r, ave_payoff_list, x); // 各タイプの割合や獲得した利得、協力率をテキストファイルに出力

        calculateFrequency(mu_new, mu, ave_payoff_list); // 獲得した適応度を用いて次世代の割合を計算

        // 微小な確率で突然変異発生
        if (uni(mt64) < mutationRate){
            mutation(x, mu_new, uni(mt64));
            cout << "mutation" << endl;
        }
        copy(mu_new, mu_new + typeN, mu);  // 配列をコピー

        // 一定の確率でランダムショック発生
        if (uni(mt64) < shockRate){
        	cout << "random shock" << endl;
        	randomShock(x, mu);
        }
    }

    outputfile.close();

    cout << "Simulation complete." << endl;
}

// 配列のコピー
vector<vector<double>> copyArray(const vector<std::vector<double>>& x_new_t) {
    vector<vector<double>> x_t = x_new_t;
    return x_t;
}

// 戦略の学習
void Learning(vector<vector<double>>& x_t, vector<vector<double>>& x_new_t, double* mu_t, vector<double>& mu_r_t){
    for (int k = 0; k < typeN; k++){
        if (mu_t[k] > 0.00001){ //集団中に存在するタイプのみ学習
            for (int r_id2 = 0; r_id2 < 8; r_id2++){
            	x_new_t[k][r_id2] = x_t[k][r_id2]+mu_r_t[r_id2]*delta*x_t[k][r_id2]*(calculateUtility(1.0,k,r_id2,x_t,mu_t,mu_r_t)-calculateUtility(x_t[k][r_id2],k,r_id2,x_t,mu_t,mu_r_t));
            }
        }
    }
}

// 各rの割合を計算
vector<double> calculateMuR(double* mu_t){
	vector<double> mu_r_t(8, 0.0);
	for (int k = 0; k < typeN; k++){
		int r1 = (k-(k%108))/108;
		int r2 = ((k%108)-(k%108)%54)/54;
		int r3 = ((k%54)-(k%54)%27)/27;
		mu_r_t[r1*4+r2*2+r3] += mu_t[k];
	}
	return mu_r_t;
}

// 期待効用の計算
double calculateUtility(double x1, int id1, int r_id2, vector<vector<double>>& x_t, double* mu_t, vector<double>& mu_r_t){
    // タイプナンバーをrとeに変換
    int r1_1 = (id1-(id1%108))/108;
    int r2_1 = ((id1%108)-(id1%108)%54)/54;
    int r3_1 = ((id1%54)-(id1%54)%27)/27;
    int e1_1 = ((id1%27)-(id1%27)%9)/9-1;
    int e2_1 = ((id1%9)-(id1%9)%3)/3-1;
    int e3_1 = id1%3-1;
    int r_id1 = r1_1*4 + r2_1*2 + r3_1;

    int r1_2 = ((r_id2%8)-(r_id2%8)%4)/4;
    int r2_2 = ((r_id2%4)-(r_id2%4)%2)/2;
    int r3_2 = r_id2%2;

    //　期待効用の計算
    double u = 0.0;
    for (int k = 0; k < 27; k++){
        int e1_2 = ((k%27)-(k%27)%9)/9-1;
        int e2_2 = ((k%9)-(k%9)%3)/3-1;
        int e3_2 = k%3-1;
        int id2 = r1_2*108+r2_2*54+r3_2*27+(e1_2+1)*9+(e2_2+1)*3+(e3_2+1);
        if (mu_r_t[r_id2] > 0.000001){
        	u += (x1*x_t[id2][r_id1]*(b-c-c_e*abs(e1_1)+r1_1*e1_2)+x1*(1-x_t[id2][r_id1])*(-c-c_e*abs(e3_1)+r3_1*e2_2)+(1-x1)*x_t[id2][r_id1]*(b-c_e*abs(e2_1)+r2_1*e3_2)+(1-x1)*(1-x_t[id2][r_id1])*0)*mu_t[id2]/mu_r_t[r_id2];
        }
    }
    return u;
}

// 平均適応度の計算
void calculateFitness(double* mu_t, vector<vector<double>>& x_t, double* ave_payoff_list_t){
    for (int k = 0; k < typeN; k++){
        if (mu_t[k] > 0.0000001){ //集団中に存在するタイプのみ計算
        	int r1_1 = (k-(k%108))/108;
        	int r2_1 = ((k%108)-(k%108)%54)/54;
        	int r3_1 = ((k%54)-(k%54)%27)/27;
            int e1_1 = ((k%27)-(k%27)%9)/9-1;
            int e2_1 = ((k%9)-(k%9)%3)/3-1;
            int e3_1 = k%3-1;
            int r_id1 = r1_1*4+r2_1*2+r3_1;

            for (int l = 0; l < typeN; l++){
            	int r1_2 = (l-(l%108))/108;
            	int r2_2 = ((l%108)-(l%108)%54)/54;
            	int r3_2 = ((l%54)-(l%54)%27)/27;
            	int r_id2 = r1_2*4+r2_2*2+r3_2;
                ave_payoff_list_t[k] += mu_t[l]*(x_t[k][r_id2]*x_t[l][r_id1]*(b+c_e*(1-abs(e1_1)))+x_t[k][r_id2]*(1-x_t[l][r_id1])*(c_e*(1-abs(e3_1)))+(1-x_t[k][r_id2])*x_t[l][r_id1]*(b+c+c_e*(1-abs(e2_1)))+(1-x_t[k][r_id2])*(1-x_t[l][r_id1])*(c+c_e));
            }
        }
    }
}

// 次世代の割合を計算
void calculateFrequency(double* new_mu_t, double* mu_t, double* ave_payoff_list_t){
    double ave_ave_fit = 0;
    for (int j = 0; j < typeN; j++) {
        ave_ave_fit += mu_t[j]*ave_payoff_list_t[j];
    }

    for (int j = 0; j < typeN; j++) {
        new_mu_t[j] = mu_t[j]*(d + ave_payoff_list_t[j])/(d + ave_ave_fit);
        ave_payoff_list_t[j] = 0; // 適応度のリセット
    }
}

// 突然変異
void mutation(vector<vector<double>>& x_t, double* mu_new_t, double rnd){
    // ランダムに一つのタイプが選ばれる
    int id = rand()%216;
    // もし選ばれたタイプが突然変異前に根絶していた場合、学習した協力率はリセット
	if(mu_new_t[id] <= 0.0000001){
		for (int l = 0; l < 8; l++){
			x_t[id][l] = 0.5;
		}
	}
    // 選ばれたタイプの割合に(0,0.1)の範囲の乱数を足し、全体が1になるように調整する
    mu_new_t[id] = mu_new_t[id] + (rnd/10.0);
    for (int j = 0; j < typeN; j++) {
        mu_new_t[j] = mu_new_t[j]/(1+rnd/10.0);
    }
}

void randomShock(vector<vector<double>>& x, double* mu){
	std::mt19937_64 mt64_t(rand());
    // 集団中に存在するタイプの中からランダムに一つが選ばれる
	int id = rand()%216;
	while (mu[id] < 0.000001){
		id = rand()%216;
	}
    // 選ばれたタイプのすべてのrに対する戦略が、切断正規分布（ショック前の戦略を平均とする）に従ってランダムに変更
	for (int k = 0; k < 8; k++) {
		std::normal_distribution<double> normal(x[id][k], 0.2);
		double number = 2.0;
		while ((number >= 1.0) or (number <= 0.0)){
			number = normal(mt64_t);
		}
		x[id][k] = number;
	}
}

// 結果の出力
void outputResult(int generation, double* mu_t, vector<double>& mu_r_t, double* ave_payoff_list_t, vector<vector<double>>& x_t){
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

        for (int k = 0; k < 8; k++) {
        	CRate[j] += mu_r_t[k]*x_t[j][k];
        }
        TotalCRate += mu_t[j]*CRate[j];
		outputfile<< generation << "\t" << j << "\t" << r1 << "\t" << r2 << "\t" << r3 << "\t" << e1 << "\t" << e2 << "\t" << e3 << "\t"<< mu_t[j]<< "\t" << ave_payoff_list_t[j] << "\t" << CRate[j] << "\t" << TotalCRate;
		outputfile<<"\n";
	}
}

