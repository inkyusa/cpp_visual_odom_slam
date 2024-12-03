


#include <cmath>
#include <iostream>
#include <vector>
using namespace std;


class GridMap {
    public:
    GridMap(int row, int col) : nRow_(row), nCol_(col) {
        map_.resize(nRow_, vector<int>(nCol_, false));
    }
    int set(int row, int col, bool val) {
        if (row < 0 or row >= nRow_ or col < 0 or col >= nCol_) {
            cout << "Input " << row << "," << col << " is out of range" << endl;
            return -1;
        }
        else {
            map_[row][col] = val;
            return true;
        }
    }
    int get(int row, int col) {
        if (row < 0 or row >= nRow_ or col < 0 or col >= nCol_) {
            cout << "Input " << row << "," << col << " is out of range" << endl;
            return -1;
        }
        else {
            bool ret = map_[row][col];
            return ret;
        }
    }

private:
    int nRow_;
    int nCol_;
    vector<vector<int>> map_;
};

int main(void) {
    GridMap* grid = new GridMap(4, 5);
    cout << grid->get(0, 3) << endl;
    cout << grid->set(0, 3, 1) << endl;
    cout << grid->get(0, 3) << endl;
    
    cout << grid->set(4, 4, 1) << endl;
    
}