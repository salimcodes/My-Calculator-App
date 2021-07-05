#include <iostream>

using namespace std;

int main()
{

    cout << "IF a set of four variable equation is given as follows: " << endl;
    cout << "Where M1, M2, M3 and M4 are four unknown variables " << endl;
    cout << "Where equation 1 is given as aM1 + bM2 + cM3 + dM4 = w " << endl;
    cout << "Where equation 2 is given as eM1 + fM2 + gM3 + hM4 = x " << endl;
    cout << "Where equation 3 is given as iM1 + jM2 + kM3 + lM4 = y " << endl;
    cout << "Where equation 4 is given as mM1 + nM2 + oM3 + pM4 = z " << endl;

    double a, b, c, d, w, e, f, g, h, x, i, j, k, l, y, m, n, o, p, z, D, D1, D2, D3, D4;
    cout << "Enter a: " << endl;
    cin >> a;
    cout << "Enter b: " << endl;
    cin >> b;
    cout << "Enter c: " << endl;
    cin >> c;
    cout << "Enter d: " << endl;
    cin >> d;
    cout << "Enter w: " << endl;
    cin >> w;
    cout << "Enter e: " << endl;
    cin >> e;
    cout << "Enter f: " << endl;
    cin >> f;
    cout << "Enter g: " << endl;
    cin >> g;
    cout << "Enter h: " << endl;
    cin >> h;
    cout << "Enter x: " << endl;
    cin >> x;
    cout << "Enter i: " << endl;
    cin >> i;
    cout << "Enter j: " << endl;
    cin >> j;
    cout << "Enter k: " << endl;
    cin >> k;
    cout << "Enter l: " << endl;
    cin >> l;
    cout << "Enter y: " << endl;
    cin >> y;
    cout << "Enter m: " << endl;
    cin >> m;
    cout << "Enter n: " << endl;
    cin >> n;
    cout << "Enter o: " << endl;
    cin >> o;
    cout << "Enter p: " << endl;
    cin >> p;
    cout << "Enter z: " << endl;
    cin >> z;
    D= (a*f*k*p)-(a*f*l*o)-(a*g*j*p)+(a*g*n*l)+(a*h*j*o)-(a*h*n*k)-(b*e*k*p)+(b*e*l*o)+(b*g*i*p)-(b*g*m*l)-(b*h*i*o)+(b*h*m*k)+(c*e*j*p)-(c*e*l*n)-(c*f*i*p)+(c*f*m*l)+(c*h*i*n)-(c*h*m*j)-(d*e*j*o)+(d*e*k*n)+(d*f*i*o)-(d*f*m*k)-(d*g*i*n)+(d*g*m*j);
    D1= (w*f*k*p)-(w*f*l*o)-(w*g*j*p)+(w*g*n*l)+(w*h*j*o)-(w*h*k*n)-(b*x*k*p)+(b*x*l*o)+(b*g*y*p)-(b*g*z*l)-(b*h*y*o)+(b*h*z*k)+(c*x*j*p)-(c*x*l*n)-(c*f*y*p)+(c*f*z*l)+(c*h*y*n)-(c*h*z*j)-(d*x*j*o)+(d*x*k*n)+(d*f*y*o)-(d*f*z*k)-(d*g*y*n)+(d*g*z*j);
    D2= (a*x*k*p)-(a*x*l*o)-(a*g*y*p)+(a*g*z*l)+(a*h*y*o)-(a*h*z*k)-(w*e*k*p)+(w*e*l*o)+(w*g*i*p)-(w*g*m*l)-(w*h*i*o)+(w*h*m*k)+(c*e*y*p)-(c*e*l*z)-(c*x*i*p)+(c*x*m*l)+(c*h*i*z)-(c*h*m*y)-(d*e*y*o)+(d*e*k*z)+(d*x*i*o)-(d*x*m*k)-(d*g*i*z)+(d*g*m*y);
    D3= (a*f*y*p)-(a*f*l*z)-(a*x*j*p)+(a*x*n*l)+(a*h*j*z)-(a*h*n*y)-(b*e*y*p)+(b*e*l*z)+(b*x*i*p)-(b*x*m*l)-(b*h*i*z)+(b*h*m*y)+(w*e*j*p)-(w*e*l*n)-(w*f*i*p)+(w*f*m*l)+(w*h*i*n)-(w*h*m*j)-(d*e*j*z)+(d*e*y*n)+(d*f*i*z)-(d*f*m*y)-(d*x*i*n)+(d*x*m*j);
    D4= (a*f*k*z)-(a*f*y*o)-(a*g*j*z)+(a*g*n*y)+(a*x*j*o)-(a*x*n*k)-(b*e*k*z)+(b*e*y*o)+(b*g*i*z)-(b*g*m*y)-(b*x*i*o)+(b*x*m*k)+(c*e*j*z)-(c*e*y*n)-(c*f*i*z)+(c*f*m*y)+(c*x*i*n)-(c*x*m*j)-(w*e*j*o)+(w*e*k*n)+(w*f*i*o)-(w*f*m*k)-(w*g*i*n)+(w*g*m*j);

    cout << "M1 is given as " << D1/D <<endl;

    cout << "M2 is given as " << D2/D <<endl;

    cout << "M3 is given as " << D3/D <<endl;

    cout << "M4 is given as " << D4/D <<endl;

    //return 0;

}
