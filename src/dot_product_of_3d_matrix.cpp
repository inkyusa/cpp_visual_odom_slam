#include <cmath>
#include <iostream>
#include <vector>
#include <optional>
using namespace std;

template<typename T>
optional<T> dotProduct (vector<T>& a, vector<T>& b) {
    if (a.size() != b.size()) {
        cout << "input vectors' dim should matched" << endl;
        return nullopt;
    }
    int n = a.size();
    T ret = 0;
    for (int i = 0; i < n; i++) {
        ret += a[i] * b[i];
    }
    return ret;

}

int main(void) {
    std::vector<double> vector1 = {1.0, 2.0, 3.0};
    std::vector<double> vector2 = {4.0, -5.0, 6.0};
    auto ret = dotProduct(vector1, vector2);
    if(ret) {
        cout << *ret << endl;
    }
    return 0;
}