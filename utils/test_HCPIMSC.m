function [R, history, X, omega, G, Z] = test_HCPIMSC(Xo, Po, Xu, Pu, S, G, lambda1, lambda2, lambda3,lambda4,lambda5,lambda6, c)
% Inputs:
%   Xo - observed parts, a cell array, num_view*1, each array is d_v*n_v
%   Po - projection matrices, a cell array, num_view*1, each array is n_v*n
%   Xu - missing parts, a cell array, num_view*1, each array is d_v*(n-n_v)
%   Pu - projection matrices, a cell array, num_view*1, each array is (n-n_v)*n
%   lambda1,lambda2 - hyperparameters for the algorithm
%   c - number of clusters
% Outputs:
%   R - unified affinity matrix, a n*n array
%   history - reconstruction errors
%   X: reconstructed samples
%   omega - weights of views
%   G - refined view-specific affinity matrices, a cell array, num_view*1, each array is n*n
%   Z - view-specific affinity matrices, a cell array, num_view*1, each array is n*n

%lambda1:lambda
%lambda2: gamma
[~,n2,~]=size(S);
for i=1:n2
    S_sss(:,:,i)=G{i}'*S{i}*G{i}; %index matrix G
    Z1{i}=S_sss(:,:,i);
    Q2{i}=zeros(size(S{i}));
    P{i}=Q2{i};
    E{i}=P{i};
end
dim=size(S_sss);
[n1,n2,n3]=size(S_sss);

%parameter initial
num_view = length(Xo);
N = size(Po{1}, 2);
%matrix initial
Z = cell(num_view, 1);
X = cell(num_view, 1);
Xc = cell(num_view, 1);
Q1=zeros(dim); Q3=cell(num_view, 1); W=zeros(dim); Y=cell(num_view, 1);
F=zeros(n2,c); Z1=zeros(dim);R_tensor=zeros(dim);M=zeros(dim);

R = zeros(N, N);
for v = 1 : num_view
    Z{v} = zeros(N, N);
    Q3{v}=zeros(N, N);
    Y{v}=zeros(N, N);
    X{v} = Xo{v} * Po{v};
    Xc{v} = Xo{v} * Po{v};
end
omega = ones(v, 1) ./ v;

mu = 1e-4;
r = lambda4;
b = lambda5; %
p=lambda6; %shcatten
mode=2;
lambda=lambda3;%beta
beta=ones(n1,1);
alpha=1/n3*ones(1,n3);
for i=1:n3
    [nu,~]=size(S{i});
    omega1{i}=G{i}'*ones(nu,nu)*G{i};%ones(numFold,numFold);
end

for iter = 1 : 31
    X_k=Z1;
    Z_k=M;
    fprintf('----processing iter %d--------\n', iter);
    % update Z
    tempZ = zeros(N, N);
    for v = 1 : num_view
        tmp = X{v}' * X{v};
        Z{v} = ((omega(v) + lambda2) * eye(N, N) + tmp)\(tmp + omega(v) * R + lambda2 * M(:,:,v));
        Z{v} = Z{v} - diag(diag(Z{v}));
        Z{v} = max(0.5 * (Z{v} + Z{v}'), 0 );
        tempZ = tempZ + omega(v) * Z{v};
    end
    tempZ = tempZ ./ sum(omega);
    R = tempZ - diag(diag(tempZ));
    R = max(0.5 * (R + R'), 0);
    R_tensor(:,:,v)=R;
    % update omega
    for v = 1 : num_view
        omega(v) = 0.5 / (norm(Z{v} - R, 'fro') + eps);
    end
    % update L
    if iter == 1
        Weight = constructW_PKN(R, 15);
        Diag_tmp = diag(sum(Weight));
        L = Diag_tmp - Weight;
    else
        param.num_view = 15; 
        HG = gsp_nn_hypergraph(R', param);
        L = HG.L;
    end
    % update Xu
    temp_M = cell(num_view, 1);
    for v = 1 : num_view
        temp_M{v} = (Z{v} - eye(N)) * (Z{v} - eye(N))' + lambda1 * L;
        Xu{v} = ( - Xo{v} * Po{v} * temp_M{v} * Pu{v}') / (Pu{v} * temp_M{v} * Pu{v}' );
        [Xu{v}] = NormalizeData(Xu{v});
        % update X
        X{v} = Xc{v} + Xu{v} * Pu{v};
    end
      %% Update Z1: Z1=1/2*J-1/2*mu*C_omega(Q_3)
    for j=1:n3
        Z1(:,:,j)=0.5*(M(:,:,j)-1/mu*Q1(:,:,j)+W(:,:,j)-1/mu*Q3{j})-1/mu*(Y{j}.*omega1{j});
    end
    
    %% Update  W: w_ij=R_ij-beta/mu*alpha*D_ij+phi/mu
    for j=1:n3
        temp1 = L2_distance_1(F',F');
        temp2 = Z1(:,:,j)+Q3{j}/mu;
        linshi_W = temp2-alpha(j)^r*b*temp1/mu;
        linshi_W = linshi_W-diag(diag(linshi_W));
        for ic = 1:size(Z1(:,:,j),2)
            ind = 1:size(Z1(:,:,j),2);
            %             ind(ic) = [];
            W(ic,ind,j) = EProjSimplex_new(linshi_W(ic,ind));%  min  1/2 || x - v||^2     %  s.t. x>=0, 1'x=1
        end
    end
    clear temp1 temp2
    
    %% Update M  *
    for j =1:n3
        Z_t(:,:,j)=Z{j};
    end
    temp_M = ((2*lambda*R_tensor+mu*Z1+Q1)/(2*lambda+mu));
    %        B=Z1+Q1/mu;
    %        Zt_1=permute(Z_t,[2 1 3]);
    %        B_1=permute(B,[2 1 3]);
    [M,~,~] = prox_tnn(temp_M,1*beta/(2*lambda+mu),p,mode);
    
    %% Update F
    temp_W=zeros(size(W(:,:,1)));
    for j=1:n3
        temp_W=alpha(j)^r*W(:,:,j)+temp_W;
    end
    temp_W = (temp_W+temp_W')/2;
    L_D = diag(sum(temp_W));
    L_Z = L_D-temp_W;
    [F, ~, ~]=eig1(L_Z, c, 0);
    
    %% Update E
    for j=1:n3
        temp1 = S{j}-P{j}+Q2{j}/mu;
        temp2 = lambda/mu;
        E{j}= max(0,temp1-temp2)+min(0,temp1+temp2);
    end
    
    clear temp1 temp2
    %% Update alpha
    
    for j=1:n3
        h(j)=trace(F'*W(:,:,j)*F);
        temp(j)=((r*h(j))^(1/(1-r)));
    end
    if sum(temp)==0
        for j=1:n3
            alpha(j)=1/n3;
        end
    elseif sum(temp)~=0
        for j=1:n3
            alpha(j)=temp(j)/sum(temp);
        end
    end
    
    %record the iteration information
    history.term1(iter) = 0;
    % coverge condition
    chgX=max(abs(Z1(:)-X_k(:)));
    chgZ=max(abs(M(:)-Z_k(:)));
    chgX_Z=max(abs(Z1(:)-M(:)));
    chg=max([chgX chgZ chgX_Z]);
    for v = 1 : num_view
        history.term1(iter) = history.term1(iter) + norm(X{v} - X{v} * Z{v}, 'fro') ^ 2 ;
    end
    obj(iter) = history.term1(iter);
     if iter > 2 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1))< 1e-4 && chg<1e-4
         break;
     end
end
end