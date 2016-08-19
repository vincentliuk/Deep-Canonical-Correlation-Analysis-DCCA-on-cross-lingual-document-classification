X = load('JW11_fold0/mfcc.train');
Y = load('JW11_fold0/xrmb.train');
size(X)
size(Y)
[Wx, Wy, r] = cca(X', Y');