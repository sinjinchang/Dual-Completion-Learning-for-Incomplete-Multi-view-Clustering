clear;
clc;
addpath('ClusteringMeasure', 'LRR', 'utils');
addpath(genpath('gspbox-0.7.0/'));
resultdir1 = 'Results/';
resultdir2 = 'totalResults/';
% datadir='./data/';
% dataname={'MSRCV1_3v'};
% numdata = length(dataname); % number of the test datasets
% numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};
datadir='./data/';
dataname = {'ORL_mtv_Per0.1'};
numdata = length(dataname); % number of the test datasets
numname = {''};
for idata = 1 : 1
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    % result = [Fscore Precision Recall nmi AR Entropy ACC Purity];
    %     for dataIndex = 1: 2: 9
    for dataIndex = 1 : 1
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        %data preparation...
        xx =  load(datafile);
        data = xx.data;
        X =data;
        gt = xx.truelabel{1};
        Se = xx.S;
        index = xx.index;
        num_view = length(data);
        cls_num = length(unique(gt));
        num_sample=size(data{1},2);
        tic;
        [X1, O1, X2, O2] = DataPreparing(data, index);
        time1 = toc;
        maxAcc = 0;
        TempLambda1 = [ 100];
        TempLambda2 = [0.1 ];
        TempLambda3 = [0.01 0.1 1 10 100]; 
        TempLambda6 = [0.1 0.3 0.5 0.7 0.9]; %p
        %         TempLambda2 = 1;
        %         TempLambda1 = 1;
        ACC = zeros(length(TempLambda1), length(TempLambda2));
        NMI = zeros(length(TempLambda1), length(TempLambda2));
        Purity = zeros(length(TempLambda1), length(TempLambda2));
        idx = 1;
        
        %%%%%%%%%%%Tensor_Completion%%%%%%%%%%%%%%%
        ind_folds=zeros(num_sample,num_view);
        for iv = 1:num_view
            ind_folds(index{iv},iv)=1;
        end
        for iv = 1:num_view
            X_temp = X{iv}';
            X_temp = NormalizeFea(X_temp,1);
            ind_0 = find(ind_folds(:,iv) == 0);
            X_temp(ind_0,:) = [];
            Y{iv} = X_temp';
            W1 = eye(size(ind_folds,1));
            W1(ind_0,:) = [];
            G_temp{iv} = W1;
        end
        %% Graph construction
        S_temp=graph_construction(Y);
        for i=1:num_view
            S{i}=G_temp{i}'*S_temp{i}*G_temp{i};
            [nu,~]=size(S_temp{i});
            omega1(:,:,i)=G_temp{i}'*ones(nu,nu)*G_temp{i};
        end
        
        %%%%%%%%%%%Tensor_Completion%%%%%%%%%%%%%%%%%%%
        for LambdaIndex1 = 1 : length(TempLambda1)
            lambda1 = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                lambda2 = TempLambda2(LambdaIndex2);
                for LambdaIndex3 = 1 : length(TempLambda3)
                    lambda3= TempLambda3(LambdaIndex3);
                             for LambdaIndex6= 1 : length(TempLambda6)
                                lambda6= TempLambda6(LambdaIndex6);
                             
                disp([char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2)]);
                tic;
                 
               [S, history, X, omega, G, Z, chg] = DCIMC(X1, O1, X2, O2, S_temp, G_temp, omega1, lambda1, lambda2,lambda3, lambda6,cls_num);
                F = SpectralClustering(S, cls_num);
                time2 = toc;
                stream = RandStream.getGlobalStream;
                reset(stream);
                MAXiter = 1000; % Maximum number of iterations for KMeans
                REPlic = 20; % Number of replications for KMeans
                tic;
                res = zeros(20, 8);
                for rep = 1 : 20
                    pY = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep, : ) = Clustering8Measure(gt, pY);
                end
                time3 = toc;
                runtime(idx) = time1 + time2 + time3 / 20;
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 7);
                NMI(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 4);
                Purity(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 8);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), ...
                    '-acc=', num2str(tempResBest(dataIndex, 7)), ...
                    '-l1=', num2str(lambda1), '-l2=', num2str(lambda2) , '-l3=', num2str(lambda3),  '-l6=', num2str(lambda6) ,'_result.mat'], ...
                    'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        if tempIndex == 7
                            newS = S;
                            newF = F;
                            newX = X;
                            newG = G;
                            newZ = Z;
                            newOmega = omega;
                            newHistory = history;
                        end
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
                             end
                end
            end
        end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, : );
        PResStd = ResStd(dataIndex, : );
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC( : ))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'newS', 'newF', 'newX', 'newG', 'newZ', 'newHistory', 'newOmega', 'PResBest', 'PResStd', 'gt');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end