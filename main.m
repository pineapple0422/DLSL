clear all
clc

addpath('./utils');
%% best parameter


%% prepare data
avg = load('/home/zm/SCDA/SCDA_avgPool.mat');
max = load('/home/zm/SCDA/SCDA_maxPool.mat');

train_feature = [avg.train_data_L31a, max.train_data_L31a];
train_feature = double(train_feature');
train_label = avg.train_label;

class_label = unique(train_label);
H = zeros(length(class_label), length(train_label));
for i = 1:length(class_label)
    ind=find(train_label==class_label(i));
    H(i, ind)=1;
end



load('/home/zm/Desktop/train_global_res_fea_format.mat');

semantic = res_fea;
class_semantic = [];
for i = 1:20:4000
    temp = res_fea(i:i+19,:);
    temp_semantic = mean(double(temp));
    class_semantic = [class_semantic; temp_semantic];
end
train_semantic = class_semantic(train_label, :);
train_semantic = train_semantic';


%test data
test = load('/home/zm/Desktop/test_global_res_fea_format.mat');
test_feature = [avg.test_data_L31a, max.test_data_L31a];
test_feature = double(test_feature');
img_label = avg.test_label;


test_semantic = test.res_fea';
txt_label = [];
for i = 1:200
    temp_label = i * ones(20,1);
    txt_label = [txt_label;temp_label];  
end


Vs = train_feature;
Ts = train_semantic;


pars.dimFeature=1024;
pars.dimSemantic=1000;
pars.dimLatent=20;

pars.lambda = 0.1;
pars.beta=0.1
pars.gamma = 0.1


%% initialize 
% initial the parameters
D1 = rand(pars.dimFeature,pars.dimLatent)-0.5;
D1 = D1 - repmat(mean(D1,1), size(D1,1),1);
D1 = D1*diag(1./sqrt(sum(D1.*D1)));

D2 = rand(pars.dimSemantic,pars.dimLatent)-0.5;
D2 = D2 - repmat(mean(D2,1), size(D2,1),1);
D2 = D2*diag(1./sqrt(sum(D2.*D2)));

U=rand(length(class_label),pars.dimLatent)-0.5;
U=U-repmat(mean(U,1),size(U,1),1);
U=U*diag(1./sqrt(sum(U.*U)));

V = [Vs ; pars.lambda*Ts; pars.beta*H];
D = [D1 ; pars.lambda*D2; pars.beta*U];
Z = learn_coefficients_noise(D,V,pars.gamma);

%% parameter settings

pars.iter=100;


%% optimization
for m = 1:pars.iter
    m
	% update D1
 	D1 = learn_basis(Vs,Z, pars.gamma);
                
	% update D2
	D2 = learn_basis(Ts,Z, pars.gamma);
    
    %update U
    U=learn_basis(H,Z,pars.gamma);
                
	% update Z
	V = [Vs ; pars.lambda*Ts; pars.beta*H];
    D = [D1 ; pars.lambda*D2; pars.beta*U];
	Z = learn_coefficients_noise(D,V,pars.gamma);
    
    %test
    Z_img = learn_coefficients_noise(D1,test_feature,pars.gamma);
    Z_sem = learn_coefficients_noise(D2,test_semantic,pars.gamma);    
    
    
    cur_map1 = mAP(Z_img', Z_sem', img_label, txt_label)
    cur_map2 = mAP(Z_sem', Z_img', txt_label, img_label)
   
%  	[ ac_v, ac_a, ac_s, ac_va, ac_as, ac_vs, ac_vas] = evaluate_all( test_feature, D1, 2, test_label, Au,testClassLabels, D1, D2, pars.gamma);
%     ACC_v = [ACC_v ac_v];
%     ACC_a= [ACC_a ac_a];
%     ACC_s= [ACC_s ac_s];
%     ACC_va = [ACC_va ac_va];
%     ACC_as= [ACC_as ac_as];
%     ACC_vs= [ACC_vs ac_vs];
%     ACC_vas = [ACC_vas ac_vas];
 end
% result.TACC_V = ACC_v;
% result.TACC_a= ACC_a;
% result.TACC_s= ACC_s;
% result.TACC_va = ACC_va;
% result.TACC_as= ACC_as;
% result.TACC_vs= ACC_vs;
% result.TACC_vas = ACC_vas;

function [acc ] = mAP(image, text, image_label, text_label)
image = single(image);
text = single(text);
dist = pdist2(image, text, 'cosine');
[dis, ord] = sort(dist, 2);
[ numcases, k ] = size(dist);
res = [];
for i = 1:numcases
    order = ord(i,:);
    p = 0.0;
    r = 0.0;
    for j =  1 :k
        if image_label(i)== text_label(order(j))
            r = r + 1;
            p = p + (r / (j + 1));
        end
    end
    if r > 0
        res = [res, p / r];
    else
        res = [res, 0];
    end

end

acc = mean(res);

end

