


#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

template <typename T>
T getDistance(const vector<T>& a, const vector<T>& b) {
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
}

int main(void) {
    vector<vector<double>> a = {{0.1, 0.2}, {1.5, 1.2}, {1.3, -0.2}};
    vector<vector<double>> b = {{1.1, 2.2}, {-1.5, 3.2}, {-1.3, 0.2}};
    for (int i = 0; i < a.size(); i++) {
        double dist = getDistance(a[i], b[i]);
        cout << dist << endl;
    }
    return 0;   
}