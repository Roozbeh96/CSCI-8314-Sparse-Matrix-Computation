%% Load data
close all
clc

dataT = readtable('ml-100k/data.csv', 'ReadVariableNames', true);% ratings data
userT = readtable('ml-100k/user.csv', 'ReadVariableNames', true);% user data


data = table2array(dataT);

% Create user-item matrix
R = zeros(max(data(:,1)), max(data(:,2)));
for i = 1:size(data,1)
    R(data(i,1), data(i,2)) = data(i,3);
end

Nuser=size(R,1);
Nmovies=size(R,2);

figure
spy(R,'c.') %1,486,126 zeros
xlabel(sprintf('movies = %d, nz = %d',size(R,2),nnz(R)),'Interpreter','latex');
ylabel(sprintf('Users = %d', size(R,1)),'Interpreter','latex');
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)


%% Test and Train dataset

filename = 'ml-100k/u1.base';
dataTrain = load(filename);

filename = 'ml-100k/u1.test';
dataTest = load(filename);

Rtrain = zeros(max(data(:,1)), max(data(:,2)));
for i = 1:size(dataTrain,1)
    Rtrain(dataTrain(i,1), dataTrain(i,2)) = dataTrain(i,3);
end

Rtest = zeros(max(data(:,1)), max(data(:,2)));
for i = 1:size(dataTest,1)
    Rtest(dataTest(i,1), dataTest(i,2)) = dataTest(i,3);
end

%% User-based(UB)


% Compute similarities between users using cosine similarity
simUB = pdist(Rtrain, 'cosine');
simUB = 1 - squareform(simUB);

figure
contourf(simUB,20,'EdgeColor','none')
cmap1 = plasma(20); % Call cbrewer and get colormap
cmap1 = flip(cmap1);         % flip the colorbar
colormap(cmap1);
C1=cmap1;
C1=[1,1,1;C1];
colormap(gca,C1)
hcb1=colorbar;
title(hcb1,'similarity(U-B)','Interpreter','Latex','FontSize',13);
hcb1.TickLabelInterpreter = 'latex';
ylabel('Users','Interpreter','Latex','FontSize',13);
xlabel('Users','Interpreter','Latex','FontSize',13);
hcb1.Label.Interpreter = 'latex';
axis vis3d
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',13)
set(gca, 'color', 'none');


% Predict ratings for users

N=1:20;
meanMAEUB=zeros();


for k=1:size(N,2)

    PrirateUB = zeros(size(Rtrain,1),size(Rtrain,2));

    K=N(1,k);
    for j=1:size(Rtrain,1)
        [IDXnonrated]=find(~Rtrain(j,:));


        for i=1:size(IDXnonrated,2)

            [IDXrated]=find(Rtrain(:,IDXnonrated(1,i)));
            if size(IDXrated,1)>=K

                [W,I] = sort(simUB(j,IDXrated),'descend');
                W = W(1,1:K);
                PrirateUB(j,IDXnonrated(1,i))=(W*Rtrain(IDXrated(I(1,1:K),1),IDXnonrated(1,i)))/sum(W);
            else
                W=simUB(j,IDXrated);
                PrirateUB(j,IDXnonrated(1,i))=(W*Rtrain(IDXrated,IDXnonrated(1,i)))/sum(W);
            end
            if isnan(PrirateUB(j,IDXnonrated(1,i)))
                PrirateUB(j,IDXnonrated(1,i))=0;
            end
        end
    end


    [r,c]=find(Rtest);

    MAEUB = zeros();

    for i=1:size(r,1)

        MAEUB(1,i) = (Rtest(r(i,1),c(i,1))-PrirateUB(r(i,1),c(i,1)))^2;

    end

    meanMAEUB(1,k) = sqrt(mean(MAEUB,2));

end

%% Item-based(IB)

% Compute similarities between items using cosine similarity
simIB = pdist(Rtrain', 'cosine');
simIB = 1 - squareform(simIB);

figure
contourf(simIB,20,'EdgeColor','none')
cmap1 = plasma(20); % Call cbrewer and get colormap
cmap1 = flip(cmap1);         % flip the colorbar
colormap(cmap1);
C1=cmap1;
C1=[1,1,1;C1];
colormap(gca,C1)
hcb1=colorbar;
title(hcb1,'similarity(I-B)','Interpreter','Latex','FontSize',13);
hcb1.TickLabelInterpreter = 'latex';
ylabel('Items','Interpreter','Latex','FontSize',13);
xlabel('Items','Interpreter','Latex','FontSize',13);
hcb1.Label.Interpreter = 'latex';
axis vis3d
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',13)
set(gca, 'color', 'none');


% Predict ratings for users

meanMAEIB=zeros();


for k=1:size(N,2)
    PrirateIB = zeros(size(Rtrain,1),size(Rtrain,2));

    K=N(1,k);


    for j=1:size(Rtrain,1)

        [IDXnonrated]=find(~Rtrain(j,:));


        for i=1:size(IDXnonrated,2)

            [IDXrated]=find(Rtrain(j,:));

            if size(IDXrated,2)>=K

                [W,I]=sort(simIB(IDXnonrated(1,i),IDXrated),'descend');
                W = W(1,1:K);

                PrirateIB(j,IDXnonrated(1,i))=(W*Rtrain(j,IDXrated(1,I(1,1:K)))')/sum(W);
            else
                W = simIB(IDXnonrated(1,i),IDXrated);
                PrirateIB(j,IDXnonrated(1,i))=(W*Rtrain(j,IDXrated)')/sum(W);
            end
            if isnan(PrirateIB(j,IDXnonrated(1,i)))
                PrirateIB(j,IDXnonrated(1,i))=0;
            end
        end
    end
    [r,c]=find(Rtest);

    MAEIB = zeros();

    for i=1:size(r,1)

        MAEIB(1,i) = (Rtest(r(i,1),c(i,1))-PrirateIB(r(i,1),c(i,1)))^2;

    end

    meanMAEIB(1,k) = sqrt(mean(MAEIB,2));

end

%%

figure
plot(N,meanMAEUB,'r','LineWidth',2,'LineStyle','-',...
    'Marker','o','MarkerSize',8,'MarkerFaceColor','r')
hold on
plot(N,meanMAEIB,'color',[0.30,0.75,0.93],'LineWidth',2,'LineStyle','-.',...
    'Marker','square','MarkerSize',8,'MarkerFaceColor',[0.30,0.75,0.93])
ylim([0.9 1.4])
xlim([1 20])
xlabel('Number of Neighbor','Interpreter','Latex');
ylabel('RMSE','Interpreter','Latex');
legend('User based','Item based','Interpreter','Latex','FontSize',20)
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',20)

%% Matrix Factorization
close all

factors = [40,50,60,70,80,90,100,120,150,180,200,250,300];

meanMAEMF=zeros();
NIter = 20;

for k= 1:size(factors,2)

    f = factors(1,k);

    X = randn(size(Rtrain,1),f);
    Y = randn(size(Rtrain,2),f);
    lambda = 0.1;

    for iter=1:NIter
        
        % update item factors
        for j = 1:size(Rtrain,2)
            % find the set of users that rated item j
            idx = find(Rtrain(:, j) ~= 0);
            % calculate the matrix Xj from user factors
            Xj = X(idx, :);
            % update item j factor using least squares
            Yj = (Xj' * Xj + lambda * eye(f)) \ (Xj' * Rtrain(idx, j));
            % update the item factor matrix Y
            Y(j, :) = Yj';
        end


        % update user factors
        for i = 1:size(Rtrain,1)
            % find the set of items that user i rated
            idx = find(Rtrain(i, :) ~= 0);
            % calculate the matrix Yi from item factors
            Yi = Y(idx, :);
            % update user i factor using least squares
            Xi = (Yi' * Yi + lambda * eye(f)) \ (Yi' * Rtrain(i, idx)');
            % update the user factor matrix X
            X(i, :) = Xi';
        end


    end

    PrirateMF = X*Y';

    [r,c]=find(Rtest);

    MAEMF = zeros();

    for i=1:size(r,1)

        MAEMF(1,i) = (Rtest(r(i,1),c(i,1))-PrirateMF(r(i,1),c(i,1)))^2;

    end

    meanMAEMF(1,k) = sqrt(mean(MAEMF,2));

end


figure
plot(factors,meanMAEMF,'k','LineWidth',2,'LineStyle','-',...
    'Marker','d','MarkerSize',8,'MarkerFaceColor','k')
% ylim([0.5 2])
xlim([40 300])
xlabel('Number of factors','Interpreter','Latex');
ylabel('RMSE','Interpreter','Latex');
legend('Matrix factorization','Interpreter','Latex','FontSize',20)
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',20)

