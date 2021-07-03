#include <iostream>

using namespace std;

int main()
{
    // Program to evaluate the determinant of a Three of Three Matrix
    cout << "If the three by three matrix A is given in the form shown below, Input your values for the respective variables" << endl;
    cout << "This program will return the determinant of the three by three matrix" << endl;
    cout << "[a  b  c]" << endl;
    cout << "[d  e  f]" << endl;
    cout << "[g  h  i]" << endl;
    double a,b,c,d,e,f,g,h,i, determinant;
    cout << "Input Value of a" << endl;
    cin >> a;
    cout << "Input Value of b" << endl;
    cin >> b;
    cout << "Input Value of c" << endl;
    cin >> c;
    cout << "Input Value of d" << endl;
    cin >> d;
    cout << "Input Value of e" << endl;
    cin >> e;
    cout << "Input Value of f" << endl;
    cin >> f;
    cout << "Input Value of g" << endl;
    cin >> g;
    cout << "Input Value of h" << endl;
    cin >> h;
    cout << "Input Value of i" << endl;
    cin >> i;
    determinant = ((a*e*i)-(a*f*h)-(b*d*i)+(b*f*g)+(c*d*h)-(c*e*g));
    cout << "The Determinant is Given as " << determinant << endl;

    //return 0;
}
