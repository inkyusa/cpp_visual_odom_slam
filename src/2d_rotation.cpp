


#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

template<typename T>
T deg2rad(const T& deg) {
    return deg * M_PI / 180;
}

template<typename T>
T rad2deg(const T& rad) {
    return rad * 180 / M_PI;
}

template <typename T>
vector<T> rotate2D(const vector<T>& a, const T& theta) {
    T th = deg2rad(theta);
    vector<T> ret(2);
    ret[0] = cos(th)*a[0] - sin(th)*a[1];
    ret[1] = sin(th)*a[0] + cos(th)*a[1];
    
    return ret;
}

int main(void) {
    vector<vector<double>> a = {{0.1, 0.2}, {1, 0}, {1.3, -0.2}};
    double theta = 45; //in deg
    for (int i = 0; i < a.size(); i++) {
        vector<double> a_rorated = rotate2D(a[i], theta);
        cout << a_rorated[0] << " " << a_rorated[1] << endl;
    }
    return 0;   
}