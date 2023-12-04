%% Data preparation 
clc, clear all, close all 
% Cargar datos 

label = readtable('breastCancerLabel.csv');
data = readtable('breastCancerX.csv');

T = [data,label];

% Preparacion datos 

new_data = data{:,:};
label_total = label{:,:};

% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = new_data(~idx,:);
labelTrain = label(~idx,:);
dataTest_val  = new_data(idx,:);
label_pre = label(idx,:);


cv_2 = cvpartition(size(dataTest_val,1),'HoldOut',0.5);
idx = cv_2.test;
% Separate to training and test data
dataVal = dataTest_val(~idx,:);
labelVal= label_pre(~idx,:);
dataTest  = dataTest_val(idx,:);
labelTest= label_pre(idx,:);

% Estandarización de los datos 

[x_e, c,s]= normalize(dataTrain);
label_e = labelTrain;

x_v = normalize(dataVal, 'center', c, 'scale', s);
label_v = labelVal;

x_t = normalize(dataTest, 'center', c, 'scale', s);
label_t=labelTest;
%% Tabla de accuracy para los distintos modelos 

accuracy_comp = table;
accuracy_comp.Modelo = {'Arbol de decisión individual','Regresion logistica', 'Clasificador ingenuo de Bayes', 'Random Forest', 'Gradient Boosting', 'Red neuronal'}';
accuracy_comp.Accuracy = -9*ones(6,1);
%% Single desision tree 
% check 
tic;
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);

% 'MaxNumCategories' 
for n=1:N
    t = fitctree(x_v,categorical(label_v{:,:}),'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end

% Entrenamiento de arbol con el numero de hojas que produce menor error 

[i,j_1]=min(err);

train_leafs_AD = leafs(j_1);

Mdl_AD= fitctree(x_e,categorical(label_e{:,:}),...
        'MinLeafSize',train_leafs_AD);
t_dt = toc;
accuracy_comp.Accuracy(1) = 1 - loss(Mdl_AD,x_t,categorical(label_t{:,:}));
fig = confusionchart(categorical(label_t{:,:}),predict(Mdl_AD,x_t));
fig.Title = 'Matriz de confusión: árbol de decisión individual ';
saveas(fig,'CM_AD.png');
%% Regresion logistica 
%check 
tic;
Lambda = logspace(-6,-0.5,11);


x_v = x_v';
x_e = x_e'; 
rng(10); % For reproducibility
CVMdl = fitclinear(x_v,categorical(label_v{:,:}),'ObservationsIn','columns','CrossVal','On',...
    'Learner','logistic','Solver','sparsa','Regularization','lasso',...
    'Lambda',Lambda,'GradientTolerance',1e-8);

ce = kfoldLoss(CVMdl);
[i,j_2]=min(ce);
Lambda_f = Lambda(j_2);

Mdl_RL = fitclinear(x_e,categorical(label_e{:,:}),'ObservationsIn','columns',...
    'Learner','logistic','Solver','sparsa','Regularization','lasso',...
    'Lambda',Lambda_f,'GradientTolerance',1e-8);

t_rl = toc;
% calificacion 
accuracy_comp.Accuracy(2) = 1 - loss(Mdl_RL,x_t,categorical(label_t{:,:}));

fig = confusionchart(categorical(label_t{:,:}),predict(Mdl_RL,x_t))
fig.Title = 'Matriz de confusión: Regresión logística ';
saveas(fig,'CM_RL.png');
%% Bayes 
% check
x_v = x_v';
x_e = x_e'; 
tic;
classNames = {'1','0'};
prior = [0.017 0.983];

kernel_op = {'box', 'epanechnikov', 'normal', 'triangle'};

for k = 1:numel(kernel_op)

CVMdl1 = fitcnb(x_v,categorical(label_v{:,:}),'ClassNames',classNames,'Prior',prior,...
    'CrossVal','on','DistributionNames' ,'kernel','Kernel',kernel_op{k});
err(k) = kfoldLoss(CVMdl1);

end

[i,j_3]=min(err);
kernel_f = kernel_op{j_3};

Mdl_BC = fitcnb(x_e,categorical(label_e{:,:}),'ClassNames',classNames,'Prior',prior,...
    'DistributionNames' ,'kernel','Kernel',kernel_f);
t_bc = toc;
accuracy_comp.Accuracy(3) = 1 - loss(Mdl_BC,x_t,categorical(label_t{:,:}))
fig = confusionchart(categorical(label_t{:,:}),categorical(predict(Mdl_BC,x_t)));
fig.Title = 'Matriz de confusión: Clasificador de Bayes ';
saveas(fig,'CM_CB.png');

%% random forest 
% check
tic;
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);
rng('default')

for l=1:numel(leafs)
t = templateTree('Reproducible',true, 'MinLeafSize',leafs(l));
Mdl = fitcensemble(x_v,categorical(label_v{:,:}),'Method','Bag','CrossVal','on','Learners',t);
err(l) = kfoldLoss(Mdl);
end

[i,j_4]=min(err);
train_leafs_RD = leafs(j_4);

t = templateTree('Reproducible',true, 'MinLeafSize',train_leafs_RD);
Mdl_RF = fitcensemble(x_e,categorical(label_e{:,:}),'Method','Bag','Learners',t);
t_rf = toc;
accuracy_comp.Accuracy(4) = 1 - loss(Mdl_RF,x_t,categorical(label_t{:,:}))
fig = confusionchart(categorical(label_t{:,:}),categorical(predict(Mdl_RF,x_t)));
fig.Title = 'Matriz de confusión: Random Forest ';
saveas(fig,'CM_RF.png');

%% gradient boosting 
% check
tic;
leafs = logspace(1,2,10);
N = numel(leafs);
err = zeros(N,1);
rng('default')

for l=1:numel(leafs)
t = templateTree('Reproducible',true, 'MinLeafSize',leafs(l));
Mdl = fitcensemble(x_v,categorical(label_v{:,:}),'Method','AdaBoostM1','CrossVal','on','Learners',t,'LearnRate',0.1);
err(l) = kfoldLoss(Mdl);
end

[i,j_5]=min(err);
train_leafs_GB = leafs(j_5);

t = templateTree('Reproducible',true, 'MinLeafSize',train_leafs_GB);
Mdl_GB = fitcensemble(x_e,categorical(label_e{:,:}),'Method','AdaBoostM1','Learners',t,'LearnRate',0.1)
t_gb = toc;
accuracy_comp.Accuracy(5) = 1 - loss(Mdl_GB,x_t,categorical(label_t{:,:}));

fig = confusionchart(categorical(label_t{:,:}),categorical(predict(Mdl_GB,x_t)));
fig.Title = 'Matriz de confusión: Gradient boosting ';
saveas(fig,'CM_GB.png');

%% red neuronal 
% check
Lambda_rn = logspace(-6,1,11);
tic;
for lm =1:numel(Lambda_rn)
Mdl = fitcnet(x_v,categorical(label_v{:,:}), 'Lambda',Lambda_rn(lm),'CrossVal','on');
err(lm) = kfoldLoss(Mdl);

end 
[i,j_6]=min(err);
lamda_train_rn = Lambda_rn(j_6);

Mdl_RN = fitcnet(x_e,categorical(label_e{:,:}), 'Lambda',lamda_train_rn);
t_rn = toc;
accuracy_comp.Accuracy(6) = 1 - loss(Mdl_RN,x_t,categorical(label_t{:,:}));
fig = confusionchart(categorical(label_t{:,:}),categorical(predict(Mdl_RN,x_t)));
fig.Title = 'Matriz de confusión: Red neuronal';
saveas(fig,'CM_RN.png');


