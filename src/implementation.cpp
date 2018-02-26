#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <set>

#include "Eigen/Core"
#include "mnistLoader.h"

using namespace Eigen;
using namespace std;

class CgVariable;

class CgFunction;

class CgData;

class CgPlus;

class CgMatmul;

// +++++++++++++++++++++++++++++++++++++++++ csv reader +++++++++++++++++++++++++++++++++++++++++
bool getContents(const string &filename, vector<vector<string>> &table, const char delimiter = ',') {
    // open file
    fstream filestream(filename);
    if (!filestream.is_open()) { return false; }

    // read file
    while (!filestream.eof()) {
        string buffer; // get a line
        filestream >> buffer;

        // separate a read string by delimiter and push it to vector
        vector<string> record;              // vector for a line string
        istringstream streambuffer(buffer); // string stream
        string token;                       // a token account string
        while (getline(streambuffer, token, delimiter)) { record.push_back(token); }

        table.push_back(record);
    }
    table.pop_back();
    return true;
}
// +++++++++++++++++++++++++++++++++++++++++ /csv reader +++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++ converter +++++++++++++++++++++++++++++++++++++
vector<VectorXd> matrix2vecOfvector(const MatrixXd &batchXm) {
    vector<VectorXd> batchXv;
    long columnsize = batchXm.cols();

    for (int i = 0; i < columnsize; ++i) {
        batchXv.push_back(batchXm.col(i));
    }
    return batchXv;
}

MatrixXd vecOfvector2matrix(const vector<VectorXd> &batchXv) {
    long vecsize = batchXv[0].size();
    unsigned long batchsize = batchXv.size();

    MatrixXd batchXm = MatrixXd::Random(vecsize, batchsize);

    for (int i = 0; i < batchsize; ++i)
        for (int j = 0; j < vecsize; ++j) {
            batchXm(j, i) = batchXv[i][j];
        }
    return batchXm;
}
// +++++++++++++++++++++++++++++++++++++++++ /converter +++++++++++++++++++++++++++++++++++++

/////////////////////////////////////////// declaration ///////////////////////////////////////////
// --------------------------------------------------------- //
class CgVariable {
private:
    string name;
    shared_ptr<CgFunction> parent_function;
    bool did;
    int row, col;
    MatrixXd data;
    MatrixXd grad;
public:
    explicit CgVariable(shared_ptr<CgFunction> pf);

    shared_ptr<CgFunction> getparent();

    MatrixXd forward();

    void backward(MatrixXd &self_grad);

    int rnum();

    int cnum();

    void cleargrads();

    void showgrad();

    void showdata();

    string getname();

    void resetter();
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgFunction {
private:
public:
    CgFunction() = default;

    virtual vector<shared_ptr<CgVariable>> getparent() {
        vector<shared_ptr<CgVariable>> ptrvec(0);
        return ptrvec;
    }

    virtual MatrixXd forward() { return MatrixXd::Random(114, 514); }

    virtual void backward(MatrixXd &child_grad) {}

    virtual int rnum() { return 114; }

    virtual int cnum() { return 514; }

    virtual void showgrad() { cout << MatrixXd::Random(19, 19) << endl; };

    virtual void showdata() { cout << MatrixXd::Random(19, 19) << endl; }

    virtual string getname() { return "YJSNPI"; }

    virtual bool istrainable() { return false; }

    virtual void update(double lr) {}
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgData : public CgFunction {
private:
    string name;
    bool train;
    int row, col;
    MatrixXd out;
    MatrixXd grad;
public:
    CgData(MatrixXd &input, int rownum, int colnum, bool trainable = false);

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    void cleargrads();

    void showgrad() override;

    void showdata() override;

    string getname() override;

    bool istrainable() override;

    void update(double lr) override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgExpand : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> par;
    int row, col;
    int smplsz;
public:
    CgExpand(shared_ptr<CgVariable> p, int sample_size);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgPlus : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> lpar, rpar;
    int row, col;
public:
    CgPlus(shared_ptr<CgVariable> lp, shared_ptr<CgVariable> rp);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgMatmul : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> wpar, xpar;
    int m, n;
public:
    CgMatmul(shared_ptr<CgVariable> wp, shared_ptr<CgVariable> xp);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgRelu : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> par;
    int row, col;
public:
    explicit CgRelu(shared_ptr<CgVariable> p);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgSigmoid : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> par;
    int row, col;
public:
    explicit CgSigmoid(shared_ptr<CgVariable> p);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //

// --------------------------------------------------------- //
class CgMSE : public CgFunction {
private:
    string name;
    shared_ptr<CgVariable> predpar, teachpar;
    int row, col;
    MatrixXd subpow;
public:
    CgMSE(shared_ptr<CgVariable> pp, shared_ptr<CgVariable> tp);

    vector<shared_ptr<CgVariable>> getparent() override;

    MatrixXd forward() override;

    void backward(MatrixXd &child_grad) override;

    int rnum() override;

    int cnum() override;

    string getname() override;
};
// --------------------------------------------------------- //
/////////////////////////////////////////// /declaration ///////////////////////////////////////////

/////////////////////////////////////////// implementation ///////////////////////////////////////////
// --------------------------------------------------------- //
CgVariable::CgVariable(shared_ptr<CgFunction> pf) {
    name = "CgVariable";
    parent_function = move(pf);
    did = false;
    row = parent_function->rnum();
    col = parent_function->cnum();
    data = MatrixXd::Random(row, col);
    grad = MatrixXd::Zero(row, col);
}

shared_ptr<CgFunction> CgVariable::getparent() { return parent_function; }

MatrixXd CgVariable::forward() {
    if (!did) {
        data = parent_function->forward();
        did = true;
    }
    return data;
}

void CgVariable::backward(MatrixXd &self_grad) {
    grad += self_grad;
    parent_function->backward(self_grad);
}

int CgVariable::rnum() { return row; }

int CgVariable::cnum() { return col; }

void CgVariable::cleargrads() { grad = MatrixXd::Zero(row, col); }

void CgVariable::showgrad() { cout << grad << endl; }

void CgVariable::showdata() { cout << data << endl; }

string CgVariable::getname() { return name; }

void CgVariable::resetter() { did = false; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgData::CgData(MatrixXd &input, int rownum, int colnum, bool trainable) {
    name = "CgData";
    train = trainable;
    row = rownum;
    col = colnum;

    // should be replaced as assert
    if (!(input.rows() == row && (input.cols() == col))) {
        cout << "bad: " << name << endl;
    }
    out = input;
    grad = MatrixXd::Zero(row, col);
}

MatrixXd CgData::forward() { return out; }

void CgData::backward(MatrixXd &child_grad) {
    grad += child_grad;
}

int CgData::rnum() { return row; }

int CgData::cnum() { return col; }

void CgData::cleargrads() { grad = MatrixXd::Zero(row, col); }

void CgData::showgrad() { cout << grad << endl; }

void CgData::showdata() { cout << out << endl; }

string CgData::getname() { return name; }

bool CgData::istrainable() { return train; }

void CgData::update(double lr) {
    out -= (lr * grad);
}
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgExpand::CgExpand(shared_ptr<CgVariable> p, int sample_size) {
    name = "CgExpand";
    par = move(p);
    row = par->rnum();
    col = sample_size;

    // should be replaced as assert
    if (par->cnum() != 1) cout << "bad: " << name << endl;
}

vector<shared_ptr<CgVariable>> CgExpand::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(par);
    return ptrvec;
}

MatrixXd CgExpand::forward() {
    MatrixXd parfwd = par->forward();
    MatrixXd expanded = MatrixXd::Zero(row, col);
    for (int c = 0; c < col; ++c)
        for (int r = 0; r < row; ++r) {
            expanded(r, c) = parfwd(r, 0);
        }
    return expanded;
}

void CgExpand::backward(MatrixXd &child_grad) {
    MatrixXd tgrad = MatrixXd::Zero(row, 1);
    for (int r = 0; r < row; ++r)
        for (int c = 0; c < col; ++c) {
            tgrad(r, 0) += child_grad(r, c);
        }
    par->backward(tgrad);
}

int CgExpand::rnum() { return row; }

int CgExpand::cnum() { return col; }

string CgExpand::getname() { return name; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgPlus::CgPlus(shared_ptr<CgVariable> lp, shared_ptr<CgVariable> rp) {
    name = "CgPlus";
    lpar = move(lp);
    rpar = move(rp);

    // should be replaced as assert
    if (!((lpar->rnum() == rpar->rnum()) && (lpar->cnum() == rpar->cnum()))) {
        cout << "bad: " << name << endl;
    }

    row = lpar->rnum();
    col = lpar->cnum();
}

vector<shared_ptr<CgVariable>> CgPlus::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(lpar);
    ptrvec.push_back(rpar);
    return ptrvec;
}

MatrixXd CgPlus::forward() { return lpar->forward() + rpar->forward(); }

void CgPlus::backward(MatrixXd &child_grad) {
    lpar->backward(child_grad);
    rpar->backward(child_grad);
}

int CgPlus::rnum() { return row; }

int CgPlus::cnum() { return col; }

string CgPlus::getname() { return name; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgMatmul::CgMatmul(shared_ptr<CgVariable> wp, shared_ptr<CgVariable> xp) {
    name = "CgMatmul";
    wpar = move(wp);
    xpar = move(xp);

    // should be replaced as assert
    if (wpar->cnum() != xpar->rnum()) {
        cout << "bad: " << name << endl;
    }

    m = wpar->rnum();
    n = xpar->cnum();
}

vector<shared_ptr<CgVariable>> CgMatmul::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(wpar);
    ptrvec.push_back(xpar);
    return ptrvec;
}

MatrixXd CgMatmul::forward() {
    return wpar->forward() * xpar->forward(); // w * x
}

void CgMatmul::backward(MatrixXd &child_grad) {
    MatrixXd wgrad = child_grad * xpar->forward().transpose();
    MatrixXd xgrad = wpar->forward().transpose() * child_grad;
    wpar->backward(wgrad);
    xpar->backward(xgrad);
}

int CgMatmul::rnum() { return m; }

int CgMatmul::cnum() { return n; }

string CgMatmul::getname() { return name; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgRelu::CgRelu(shared_ptr<CgVariable> p) {
    name = "CgRelu";
    par = move(p);
    row = par->rnum();
    col = par->cnum();
}

vector<shared_ptr<CgVariable>> CgRelu::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(par);
    return ptrvec;
}

MatrixXd CgRelu::forward() {
    return (par->forward()).unaryExpr([](double x) { return ((x > 0) ? x : 0); });
}

void CgRelu::backward(MatrixXd &child_grad) {
    MatrixXd tgrad = MatrixXd::Random(row, col);
    MatrixXd inputX = par->forward();
    for (int i = 0; i < par->rnum(); ++i)
        for (int j = 0; j < par->cnum(); ++j) {
            if (inputX(i, j) <= 0) tgrad(i, j) = 0;
            else tgrad(i, j) = child_grad(i, j);
        }
    par->backward(tgrad);
}

int CgRelu::rnum() { return row; }

int CgRelu::cnum() { return col; }

string CgRelu::getname() { return name; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgSigmoid::CgSigmoid(shared_ptr<CgVariable> p) {
    name = "CgSigmoid";
    par = move(p);
    row = par->rnum();
    col = par->cnum();
}

vector<shared_ptr<CgVariable>> CgSigmoid::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(par);
    return ptrvec;
}

MatrixXd CgSigmoid::forward() {
    return (par->forward()).unaryExpr([](double x) { return (1.0 / (1 + exp(-x))); });
}

void CgSigmoid::backward(MatrixXd &child_grad) {
    MatrixXd tgrad = MatrixXd::Random(row, col);
    MatrixXd fwdoutY = (par->forward()).unaryExpr([](double x) { return (1.0 / (1 + exp(-x))); });
    for (int i = 0; i < par->rnum(); ++i)
        for (int j = 0; j < par->cnum(); ++j) {
            tgrad(i, j) = child_grad(i, j) * fwdoutY(i, j) * (1 - fwdoutY(i, j));
        }
    par->backward(tgrad);
}

int CgSigmoid::rnum() { return row; }

int CgSigmoid::cnum() { return col; }

string CgSigmoid::getname() { return name; }
// --------------------------------------------------------- //

// --------------------------------------------------------- //
CgMSE::CgMSE(shared_ptr<CgVariable> pp, shared_ptr<CgVariable> tp) {
    name = "CgMSE";
    predpar = move(pp);
    teachpar = move(tp);

    // should be replaced as assert
    if (!((predpar->rnum() == teachpar->rnum()) && (predpar->cnum() == teachpar->cnum()))) {
        cout << "bad: " << name << endl;
    }

    row = 1;
    col = 1;
}

vector<shared_ptr<CgVariable>> CgMSE::getparent() {
    vector<shared_ptr<CgVariable>> ptrvec(0);
    ptrvec.push_back(predpar);
    ptrvec.push_back(teachpar);
    return ptrvec;
}

MatrixXd CgMSE::forward() {
    subpow = (predpar->forward() - teachpar->forward()).unaryExpr([](double x) { return pow(x, 2.0); });

    double last = 0;

    for (int i = 0; i < subpow.cols(); ++i) {
        VectorXd tmp = (VectorXd) (subpow.col(i));
        last += tmp.sum() / subpow.rows();
    }

    MatrixXd lastMat = MatrixXd::Random(row, col);
    lastMat(0, 0) = last / subpow.cols();
    return lastMat;
}

void CgMSE::backward(MatrixXd &child_grad) {
    MatrixXd tgrad = MatrixXd::Random(subpow.rows(), subpow.cols());
    vector<double> base(0);

    for (int i = 0; i < subpow.cols(); ++i) {
        VectorXd tmp = (VectorXd) subpow.col(i);
        base.push_back(sqrt(tmp.sum()));
    }

    for (int r = 0; r < subpow.rows(); ++r)
        for (int c = 0; c < subpow.cols(); ++c) {
            tgrad(r, c) = child_grad(0, 0) * 2 * (predpar->forward()(r, c) - teachpar->forward()(r, c)) /
                          (subpow.rows() * subpow.cols());
        }

    predpar->backward(tgrad);
    teachpar->backward(tgrad);
}

int CgMSE::rnum() { return row; }

int CgMSE::cnum() { return col; }

string CgMSE::getname() { return name; }
// --------------------------------------------------------- //
/////////////////////////////////////////// /implementation ///////////////////////////////////////////

// +++++++++++++++++++++++++++++++++++++++++ optimizer +++++++++++++++++++++++++++++++++++++++++
class CgOptimizer {
private:
    double lr;
    set<shared_ptr<CgVariable>> variable_ptr_set;
    set<shared_ptr<CgFunction>> function_ptr_set;
public:
    void setup(const shared_ptr<CgVariable> target) {
        queue<shared_ptr<CgVariable>> quevar;
        queue<shared_ptr<CgFunction>> quefnc;

        quevar.push(target);

        int cnt = 0;
        while (!quefnc.empty() || !quevar.empty()) {
            if (cnt % 2 == 0) { // queue variable
                while (!quevar.empty()) {
                    shared_ptr<CgVariable> chilptr = quevar.front();
                    string chilnm = chilptr->getname();
                    if (chilnm == "CgVariable") { variable_ptr_set.insert(chilptr); }
                    shared_ptr<CgFunction> parptr = chilptr->getparent();
                    quefnc.push(parptr);
                    quevar.pop();
                }
            } else { // queue function
                while (!quefnc.empty()) {
                    shared_ptr<CgFunction> chilptr = quefnc.front();
                    string chilnm = chilptr->getname();
                    bool chiltr = chilptr->istrainable();
                    if (chilnm == "CgData" && chiltr) {
                        function_ptr_set.insert(chilptr);
                    } else {
                        vector<shared_ptr<CgVariable>> vecparptr = chilptr->getparent();
                        for (const auto ptr: vecparptr) { quevar.push(ptr); }
                    }
                    quefnc.pop();
                }
            }
            cnt += 1;
        }
    }

    CgOptimizer() {}

    void lrset(double learningrate) {
        lr = learningrate;
    }

    void lrdecrease(double decratio) {
        lr *= decratio;
    }


    void printer() {
        cout << function_ptr_set.size() << endl;
        for (const auto ptr: function_ptr_set) {
            cout << endl;
            ptr->showgrad();
            cout << endl;
        }

        cout << endl;

        cout << variable_ptr_set.size() << endl;
        for (auto ptr: variable_ptr_set) {
            cout << endl;
            ptr->showdata();
            cout << endl;
        }
    }

    void variable_resetter() {
        for (const auto ptr: variable_ptr_set) {
            ptr->resetter();
            ptr->cleargrads();
        }
    }

    void update() {
        for (const auto ptr: function_ptr_set) {
            ptr->update(lr);
        }
    }

};
// +++++++++++++++++++++++++++++++++++++++++ /optimizer +++++++++++++++++++++++++++++++++++++++++

struct domcod {
    double dom;
    double cod;
};

// +++++++++++++++++++++++++++++++++++++++++ data +++++++++++++++++++++++++++++++++++++++++
vector<domcod> getdata() {
    int n;
    vector<domcod> data(0);

    string str;
    ifstream ifs("data.txt");

    if (ifs.fail()) {
        cerr << "Failed to load the data!" << endl;
    }

    getline(ifs, str);

    while (getline(ifs, str)) {
        cout << str << endl;
        for (unsigned int i = 0; i < str.size(); ++i) {
            if (str[i] == ' ') {
                double x = stod(str.substr(0, i));
                double y = stod(str.substr(i + 1, str.size() - (i + 1)));

                domcod tmpdomcod = {x, y};

                data.push_back(tmpdomcod);
            }
        }
    }

    return data;
}
// +++++++++++++++++++++++++++++++++++++++++ /data +++++++++++++++++++++++++++++++++++++++++


int main(void) {
    vector<domcod> data = getdata();

    const unsigned long sample_size = data.size();

    const int input_dim = 1;
    const int layer1st_dim = 120;
    const int layer2nd_dim = 130;
    const int layer3rd_dim = 110;
    const int output_dim = 1;

    // shaping input data
    vector<VectorXd> inputs(0);
    for (int i = 0; i < sample_size; ++i) {
        VectorXd tmpVec = VectorXd::Zero(input_dim);
        tmpVec(0) = data[i].dom;
        inputs.push_back(tmpVec);
    }
    MatrixXd input_data = vecOfvector2matrix(inputs);

    // shaping teacher data
    vector<VectorXd> outputs(0);
    for (int i = 0; i < sample_size; ++i) {
        VectorXd tmpVec = VectorXd::Zero(input_dim);
        tmpVec(0) = data[i].cod;
        outputs.push_back(tmpVec);
    }
    MatrixXd output_data = vecOfvector2matrix(outputs);


    // --------------------------------------------------------- //
    shared_ptr<CgData> X(new CgData(input_data, input_dim, sample_size));
    shared_ptr<CgVariable> Xp(new CgVariable(X));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    shared_ptr<CgData> Y(new CgData(output_data, output_dim, sample_size));
    shared_ptr<CgVariable> Yp(new CgVariable(Y));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    MatrixXd W1_data = MatrixXd::Random(layer1st_dim, input_dim);
    shared_ptr<CgData> W1(new CgData(W1_data, layer1st_dim, input_dim, true));
    shared_ptr<CgVariable> W1p(new CgVariable(W1));

    shared_ptr<CgMatmul> W1xX(new CgMatmul(W1p, Xp));
    shared_ptr<CgVariable> W1xXp(new CgVariable(W1xX));

    shared_ptr<CgRelu> Relu_W1xX_(new CgRelu(W1xXp));
    shared_ptr<CgVariable> fstOutp(new CgVariable(Relu_W1xX_));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    MatrixXd W2_data = MatrixXd::Random(layer2nd_dim, layer1st_dim);
    shared_ptr<CgData> W2(new CgData(W2_data, layer2nd_dim, layer1st_dim, true));
    shared_ptr<CgVariable> W2p(new CgVariable(W2));

    shared_ptr<CgMatmul> W2xfstOut(new CgMatmul(W2p, fstOutp));
    shared_ptr<CgVariable> W2xfstOutp(new CgVariable(W2xfstOut));

    MatrixXd b2_data = MatrixXd::Random(layer2nd_dim, 1);
    shared_ptr<CgData> b2(new CgData(b2_data, layer2nd_dim, 1, true));
    shared_ptr<CgVariable> b2p(new CgVariable(b2));

    shared_ptr<CgExpand> b2expd(new CgExpand(b2p, sample_size));
    shared_ptr<CgVariable> b2expdp(new CgVariable(b2expd));

    shared_ptr<CgPlus> W2xfstOut_add_b2expd(new CgPlus(W2xfstOutp, b2expdp));
    shared_ptr<CgVariable> W2xfstOut_add_b2expdp(new CgVariable(W2xfstOut_add_b2expd));

    shared_ptr<CgSigmoid> Sigmoid_W2xfstOut_add_b2expdp_(new CgSigmoid(W2xfstOut_add_b2expdp));
    shared_ptr<CgVariable> sndOutp(new CgVariable(Sigmoid_W2xfstOut_add_b2expdp_));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    MatrixXd W3_data = MatrixXd::Random(layer3rd_dim, layer2nd_dim);
    shared_ptr<CgData> W3(new CgData(W3_data, layer3rd_dim, layer2nd_dim, true));
    shared_ptr<CgVariable> W3p(new CgVariable(W3));

    shared_ptr<CgMatmul> W3xsndOut(new CgMatmul(W3p, sndOutp));
    shared_ptr<CgVariable> W3xsndOutp(new CgVariable(W3xsndOut));

    shared_ptr<CgRelu> Relu_W3xsndOut_(new CgRelu(W3xsndOutp));
    shared_ptr<CgVariable> trdOutp(new CgVariable(Relu_W3xsndOut_));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    MatrixXd Wlast_data = MatrixXd::Random(output_dim, layer3rd_dim);
    shared_ptr<CgData> Wlast(new CgData(Wlast_data, output_dim, layer3rd_dim, true));
    shared_ptr<CgVariable> Wlastp(new CgVariable(Wlast));

    shared_ptr<CgMatmul> WlastxtrdOut(new CgMatmul(Wlastp, trdOutp));
    shared_ptr<CgVariable> WlastxtrdOutp(new CgVariable(WlastxtrdOut));
    // --------------------------------------------------------- //

    // --------------------------------------------------------- //
    shared_ptr<CgMSE> mse(new CgMSE(WlastxtrdOutp, Yp));
    shared_ptr<CgVariable> msep(new CgVariable(mse));
    // --------------------------------------------------------- //


    CgOptimizer opt = CgOptimizer();
    opt.lrset(0.0000001);
    opt.setup(msep);

    for (int i = 0; i < 10000; ++i) {
        if (i % 1000 == 0) opt.lrdecrease(0.5);
        MatrixXd mseval = msep->forward();
        cout << "+" << mseval << "+" << i << endl;
        if (mseval(0, 0) < 1.0) break;

        MatrixXd grad = MatrixXd::Ones(1, 1);
        msep->backward(grad);

        opt.update();
        opt.variable_resetter();
    }

    MatrixXd predict = WlastxtrdOutp->forward();

    cout << predict << endl;

    vector<VectorXd> predict_vec = matrix2vecOfvector(predict);

    for (int i = 0; i < predict_vec.size(); ++i) {
        cout << predict_vec[i] << " : " << outputs[i] << endl;
    }

    return 0;
}
