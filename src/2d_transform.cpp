


#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

double deg2rad(double deg) {
    return deg / 180 * M_PI;
}
vector<double> tranform2D(const vector<double>& a, const double& theta, const vector<double>& local_frame) {
    //rotation
    double theta_rad = deg2rad(theta);
    double a_x = cos(theta_rad) * a[0] - sin(theta_rad) * a[1] + local_frame[0];
    double a_y = sin(theta_rad) * a[0] + sin(theta_rad) * a[0] + local_frame[1];
    return {a_x, a_y};
}
int main(void) {
    vector<vector<double>> a_locals = {{0.1, 0.2}, {1, 0}, {1.3, -0.2}}; //points in local frame.
    vector<double> local_frame = {1.0, 1.5};
    double theta = 45; //in deg
    for (const auto& a_local : a_locals) {
        auto a_global = tranform2D(a_local, theta, local_frame);
        cout << "a_global x = " << a_global[0] << " a_global y = " << a_global [1] << endl;
    }
    return 0;
}