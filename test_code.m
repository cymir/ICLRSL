clear;clc; close all;
fprintf('start\n');
%% loading
load 'FRGC(Face Recognition Grand Challenge).mat';
trfea = Train_DAT;
trfea = double(trfea);
trgnd = gnd;
trfea = trfea./ repmat(sqrt(sum(trfea.*trfea)),[size(trfea,1) 1]); % normalization
ttfea = Test_DAT;
ttfea = double(ttfea);
ttgnd = gnd;
ttfea = ttfea./ repmat(sqrt(sum(ttfea.*ttfea)),[size(ttfea,1) 1]); % normalization


%% training
[ W,Q,R ] = my_code(trfea,trgnd,0.1,1e-5,10);
%% testing
trfea_proj = R*trfea;
ttfea_proj = R*ttfea;
trfea_proj = trfea_proj./ repmat(sqrt(sum(trfea_proj.*trfea_proj)),[size(trfea_proj,1) 1]); % normalization
ttfea_proj = ttfea_proj./ repmat(sqrt(sum(ttfea_proj.*ttfea_proj)),[size(ttfea_proj,1) 1]); % normalization
mdl = ClassificationKNN.fit(trfea_proj',trgnd,'NumNeighbors',1);
predict_label = predict(mdl, ttfea_proj');
accuracy =length(find(predict_label == ttgnd))/length(ttgnd)*100;
