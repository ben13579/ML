#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<math.h>
// #include "matplotlib.h"

using namespace std;
using matrix=vector<vector<long double>>;

void printmatrix(matrix& A){
    for(int i=0;i<A.size();i++){
        for(int j=0;j<A[0].size();j++){
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
    }
}

matrix operator+(const matrix& A,const matrix& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    matrix result(rows, vector<long double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

matrix operator-(const matrix& A,const  matrix& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    matrix result(rows, vector<long double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }

    return result;
}

matrix operator*(const matrix& A,const  matrix& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();
    matrix result(rowsA, vector<long double>(colsB));
    if(A[0].size()!=B.size()){
        cout<<"error";
    }
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            result[i][j] = 0;
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

matrix operator*(const matrix& A, long double scalar) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    matrix result(rows, vector<long double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * scalar;
        }
    }

    return result;
}

matrix transpose(const matrix& A){
    int n=A.size(),m=A[0].size();
    matrix re(m,vector<long double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            re[i][j]=A[j][i];
        }
    }
    return re;
}

void LUdecomposition(matrix& A,matrix& L,matrix& U){
    int n=A.size();
    for(int i=0;i<n;i++){
        L[i][i]=1;
    }
    for(int i=0;i<n;i++){
        for(int j=i;j<n;j++){
            long double sum=0;
            for(int k=0;k<i;k++){
                sum+=L[i][k]*U[k][j];
            }
            U[i][j]=A[i][j]-sum;
        }
        for(int j=i+1;j<n;j++){
            long double sum=0;
            for(int k=0;k<i;k++){
                sum+=L[j][k]*U[k][i];
            }
            L[j][i]=(A[j][i]-sum)/U[i][i];
        }
    }
}

matrix Linverse(matrix& L){
    int n=L.size();
    matrix iL(n,vector<long double>(n,0));
    for(int i=0;i<n;i++){
        iL[i][i]=1;
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            long double sum=0;
            for(int k=j;k<i;k++){
                sum+=L[i][k]*iL[k][j];
            }
            iL[i][j]=-sum;
        }
    }
    return iL;
}

matrix Uinverse(matrix& U){
    int n=U.size();
    matrix iU(n,vector<long double>(n,0));
    for(int i=0;i<n;i++){
        iU[i][i]=1/U[i][i];
    }
    for(int i=n-2;i>=0;i--){
        for(int j=i+1;j<n;j++){
            long double sum=0;
            for(int k=i+1;k<=j;k++){
                sum+=U[i][k]*iU[k][j];
            }
            iU[i][j]=-sum/U[i][i];
        }
    }
    return iU;
}


matrix inverse(matrix& A){
    int n=A.size();
    matrix L(n,vector<long double>(n,0)),U(n,vector<long double>(n,0));
    LUdecomposition(A,L,U);
    // printmatrix(L);
    
    matrix iL=Linverse(L);
    matrix iU=Uinverse(U);
    // printmatrix(iL);
    return iU*iL;
}

// long double Sum(const matrix& A){
//     long double sum=0;
//     for(int i=0;i<A.size();i++){
//         for(int j=0;j<A[0].size();j++){
//             sum+=A[i][j];
//         }
//     }
//     return sum;
// }

matrix LSE(matrix&data,int n,float lambda){
    matrix A(data.size(),vector<long double>(n));
    matrix b(data.size(),vector<long double>(1));
    
    for(int i=0;i<data.size();i++){
        b[i][0]=data[i][1];
        for(int j=0;j<n;j++){
            A[i][j]=pow(data[i][0],j);
        }
    }
    
    matrix tA=transpose(A);
    // printmatrix(tA);
    matrix M=tA*A;
    // printmatrix(M);
    for(int i=0;i<M.size();i++){
        M[i][i]+=lambda;
    }
    matrix iM=inverse(M);
    // printmatrix(iM);
    matrix M1=tA*b;
    // printmatrix(M1);
    return iM*M1;
}

matrix Newton(matrix data,int n){
    matrix A(data.size(),vector<long double>(n));
    matrix b(data.size(),vector<long double>(1));  
    for(int i=0;i<data.size();i++){
        b[i][0]=data[i][1];
        for(int j=0;j<n;j++){
            A[i][j]=pow(data[i][0],j);
        }
    }

    matrix eps(1,vector<long double>(1,100));
    matrix xn1,xn(n,vector<long double>(1,0));
    while(eps[0][0]>1e-3){
        matrix t2A=transpose(A)*2;
        matrix B=t2A*A;
        xn1=xn-inverse(B)*(B*xn-t2A*b);
        eps=transpose(xn1-xn)*(xn1-xn)*(1/n);
        xn=xn1;
    }
    return xn1;
}

matrix Gradient(matrix& data,int n,float lambda){
    long double L=1e-5;
    matrix A(data.size(),vector<long double>(n));
    matrix b(data.size(),vector<long double>(1));
    matrix re(n,vector<long double>(1,0));
    for(int i=0;i<data.size();i++){
        b[i][0]=data[i][1];
        for(int j=0;j<n;j++){
            A[i][j]=pow(data[i][0],j);
        }
    }
    
    int epochs=10000;
    for(int i=0;i<epochs;i++){
        matrix y_pred=A*re;
        matrix gradient(n, vector<long double>(1, 0));
        matrix diff=y_pred-b;
        // for(int j=0;j<n;j++){
        //     for(int k=0;k<y_pred.size();k++){
        //         gradient[j][0] += diff[k][0]*A[k][j];
        //     }
        // }
        gradient = transpose(A)*diff;
        // gradient = gradient*(1/n);
        for(int j=0;j<n;j++){
            if(re[j][0]>0){
                re[j][0]-=L*(gradient[j][0]+lambda);
                // if(re[j][0]<0){
                //     re[j][0]=0;
                // }
            }else if(re[j][0]<0){
                re[j][0]-=L*(gradient[j][0]-lambda);
                // if(re[j][0]>0){
                //     re[j][0]=0;
                // }
            }else{
                if(re[j][0]-L*gradient[j][0]>L*lambda){
                    re[j][0]-=L*(gradient[j][0]+lambda);
                }else if(re[j][0]-L*gradient[j][0]<L*(-lambda)){
                    re[j][0]-=L*(gradient[j][0]-lambda);
                }else{
                    re[j][0]=0;
                }
            }
            // re[j][0]-=L*(gradient[j][0]+regularization);
        }
        // if (i  == 0) {
            // long double loss = 0;
            // for (int i = 0; i < data.size(); i++) {
            //     long double error = y_pred[i][0] - b[i][0];
            //     loss += abs(error);  // L1 norm 的損失
            // }
            // cout << "Epoch " << i << ", L1 Loss: " << loss << endl;
        //     printmatrix(gradient);
        // }
    }
    return re;
}



int main(){
    string xydata;
    ifstream fp;
    int n;
    float lambda;
    matrix v;
    stringstream ss;
    fp.open("testfile.txt");
    if (!fp.is_open()) {      // 檢查檔案是否成功打開
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    while (getline(fp,xydata)) {
        vector<long double> xy;
        ss.str("");
        ss.clear();
        ss<<xydata;
        string tem;
        while(getline(ss,tem,',')){
            long double value= stold(tem);
            // cout << value << "\n";  
            xy.push_back(value);
        }
        v.push_back(xy);
    }
    fp.close();
    cin>> n >> lambda;
    matrix ans1=LSE(v,n,lambda);
    // cout<<"LSE"<<"\n";
    // for(int i=0;i<ans1.size();i++){
    //     cout<<ans1[i][0]<<"\n";
    // }
    matrix ans2=Newton(v,n);
    // cout<<"Newton"<<"\n";
    // for(int i=0;i<ans2.size();i++){
    //     cout<<ans2[i][0]<<"\n";
    // }
    matrix ans3=Gradient(v,n,lambda);

    ofstream out;
    out.open("LSE.txt",ios::trunc);
    if (!out.is_open()) {      // 檢查檔案是否成功打開
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    for(int i=0;i<ans1.size();i++){
        // cout<<ans1[i][0]<<"\n";
        out<<ans1[i][0]<<"\n";
    }
    out.close();
    
    out.open("Newton.txt",ios::trunc);
    if (!out.is_open()) {      // 檢查檔案是否成功打開
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    for(int i=0;i<ans2.size();i++){
        // cout<<ans2[i][0]<<"\n";
        out<<ans2[i][0]<<"\n";
    }
    out.close();

    out.open("steepest.txt",ios::trunc);
    if (!out.is_open()) {      // 檢查檔案是否成功打開
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    for(int i=0;i<ans3.size();i++){
        // cout<<ans3[i][0]<<"\n";
        out<<ans3[i][0]<<"\n";
    }
    out.close();
    return 0;
}