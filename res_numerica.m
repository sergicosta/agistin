clear all
clc

A = 900;
B = 6;
Ho = 305;
K = 0.105;
c = 100;

Q_total = 15;

cost=[1;5];

cost_fun = @(p,i) p.*cost(i);
P_fun = @(q,h) q.*h.*1e3*9.81;
H_fun = @(q) A-B.*q.^2;
Q_fun = @(a) sqrt((A-Ho)./(1./a.*K.*c-B));
a_vector = linspace(0,1,100);

Q_res = zeros(length(cost),length(a_vector));
H_res = zeros(length(cost),length(a_vector));
P_res = zeros(length(cost),length(a_vector));
cost_res = zeros(length(cost),length(a_vector));

for ii=1:length(cost)
    Q_res(ii,:) = Q_fun(a_vector);
    H_res(ii,:) = H_fun(Q_res(ii,:));
    P_res(ii,:) = P_fun(Q_res(ii,:), H_res(ii,:));
    cost_res(ii,:) = cost_fun(P_res(ii,:),ii);
end

cost_res_sum = zeros(length(a_vector));
Q_res_sum = zeros(length(a_vector));
a_mat = zeros(length(a_vector));
for ii=1:length(a_vector)
    for jj=1:length(a_vector)
        cost_res_sum(ii,jj) = cost_res(1,ii)+cost_res(2,jj);
        Q_res_sum(ii,jj)= Q_res(1,ii)+Q_res(2,jj);
    end
end

Q_res_sum_total = abs(Q_res_sum-Q_total)<=0.1;

[mr,mir] = min(cost_res_sum./Q_res_sum_total);
[mc,mic] = min(mr);
[mic, mir(mic)];

pcolor(cost_res_sum)
hold on
colorbar
plot(mic,mir(mic),'ro','MarkerSize',5 , 'MarkerFaceColor','red')
hold off

figure
pcolor(cost_res_sum.*Q_res_sum_total)
hold on
plot(mic,mir(mic),'ro','MarkerSize',5 , 'MarkerFaceColor','red')
colorbar
hold off


