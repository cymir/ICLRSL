function [ W,Q,R ] = my_code( X,X_label,lambda1,lambda2,lambda3)
%% ------- ---º¯Êý½éÉÜ----------
% W:projection matrix from feature subspace to label space
% Q:projection matrix from data subspace to feature subspace
% R:the introduced auxiliary matrix, R = WQ'
% X:data matrix of training samples; X_label:label vector of training samples
% lambda1,lambda2,lambda3,£ºparameters
%% -----------³õÊ¼»¯------------
Max_iter = 10;
[image_size,image_num] = size(X);
mu = 0.1; 
rho = 1.01;
max_mu = 10^6;
class_num = length(unique(X_label));
dim_size = 2*class_num; % subspace dimension
Y = Pre_label(X_label); % construct label matrix

W = zeros(class_num,dim_size); % c x d
Q = rand(image_size,dim_size); % m x d
R = zeros(class_num,image_size); % c x m
D = zeros(dim_size,image_num); % d x n
T1 = zeros(class_num,image_size);
T2 = zeros(dim_size,image_num);
xx = X*X';
%% -----------start iteration------------
for iter = 1:Max_iter
    % update R
    R = (Y*X'+mu*(W*Q'+T1/mu))/(xx+mu*eye(image_size));
    % update W
    W = mu*(R-T1/mu)*Q/(lambda3*eye(dim_size)+mu*Q'*Q);
    % update Q
    v  = sqrt(sum(Q.*Q,2)+eps);
    temp_q = diag(1./(v));
    a_l = lambda2*temp_q + mu*xx;
    b_l = mu*W'*W;
    c_l = mu*X*D'+mu*R'*W-X*T2'-T1'*W;
    Q = sylvester(a_l,b_l,c_l);
    
    % update D
    temp_D = [];
    rank_di = 0;
    for i = 1:class_num
        index_i = find(X_label == i);
        X_i = X(:,index_i);
        T2_i = T2(:,index_i);
        tau_d = lambda1/mu;
        temp_d = Q'*X_i+T2_i/mu;
        [uu_s,ss_s,vv_s] = svd(temp_d,'econ');
        ss_s = diag(ss_s);
        SVP = length(find(ss_s>tau_d));
        if SVP>=1
            ss_s = ss_s(1:SVP)-tau_d;
        else
            SVP = 1;
            ss_s = 0;
        end
        D_i = uu_s(:,1:SVP)*diag(ss_s)*vv_s(:,1:SVP)';
        temp_D = [temp_D,D_i];
        rank_di = rank_di+rank(D_i);
    end
    D = temp_D;
    % update T1£¬T2£¬mu
    T1 = T1+mu*(W*Q'-R);
    T2 = T2+mu*(Q'*X-D);
    mu = min(rho*mu,max_mu);
    
    leq1 = 0.5*norm(Y-W*Q'*X,'fro').^2;
    leq2 = lambda1*rank_di;
    leq3 = lambda2*sum(v);
    leq4 = 0.5*lambda3*norm(W,'fro').^2;
    obj(iter) = leq1 + leq2 + leq3 + leq4;
%     RR{iter} = R;
%     fprintf('iter:%d  ',iter);
%     fprintf('obj:%.2f\n',obj(iter));
    if iter > 2
        if abs(obj(iter)-obj(iter-1)) < 1e-2
            break
        end
    end
end
