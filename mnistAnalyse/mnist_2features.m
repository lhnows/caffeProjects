%plot(VarName1,VarName2,'*r')

n(1:10) =0

for i=1 : 10000
    index = VarName6(i)+1;
    n(index) = n(index) +1;
    x{index}(n(index)) = VarName4(i);
    y{index}(n(index)) = VarName5(i);
end;

figure(1);
plot(x{1},y{1},'.','Color',[0.3 0.0 0.0]);
hold on;
plot(x{2},y{2},'.','Color',[0.6 0.3 0.4]);
hold on;
plot(x{3},y{3},'.','Color',[0.9 0.4 0.5]);
hold on;
plot(x{4},y{4},'.','Color',[0.4 0.3 0.0]);
hold on;
plot(x{5},y{5},'.','Color',[0.4 0.6 0.0]);
hold on;
plot(x{6},y{6},'.','Color',[0.6 0.9 0.0]);
hold on;
plot(x{7},y{7},'.','Color',[0.7 0.8 0.0]);
hold on;
plot(x{8},y{8},'.','Color',[0.8 0.9 0.5]);
hold on;
plot(x{9},y{9},'.','Color',[0.9 1.0 0.8]);
hold on;
plot(x{10},y{10},'.','Color',[0.2 0.3 0.4]);