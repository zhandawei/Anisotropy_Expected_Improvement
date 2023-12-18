% Dawei Zhan, Yuqian Gui, Tianrui Li. An Anisotropic Expected Improvement 
% Criterion for Kriging-Assisted Evolutionary Computation. IEEE Congress 
% on Evolutionary Computation, 2023.
clearvars;clc;close all;
% test problem
fun_name = 'Ellipsoid';
num_vari = 50;
max_evaluation = 1000;
num_initial = 200;
num_q = 1;
pop_size = 50;
CR = 0.8;
F = 0.8;
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of current generation
generation = 1;
% generate random samples
sample_x = lhsdesign(num_initial, num_vari,'criterion','maximin','iterations',1000).*(upper_bound - lower_bound) + lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation =  size(sample_x,1);
% best objectives in each generation
fmin_record = zeros(max_evaluation - evaluation + 1,1);
% the first DE population
[~,index] = sort(sample_y);
pop_vari = sample_x(index(1:pop_size),:);
pop_obj = sample_y(index(1:pop_size),:);
fmin = sample_y(index(1));
xmin = sample_x(index(1),:);
fmin_record(generation,:) = fmin;
% print the iteration information
fprintf('KAEA-AEI on %d-D %s, generation: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,generation,evaluation,fmin);
kriging_model = Kriging_Train(sample_x,sample_y,lower_bound,upper_bound,1,0.000001,100);
% the evoluation of the population
while evaluation < max_evaluation
    % mutation
    pop_mutation = zeros(pop_size,num_vari);
    for ii = 1 : pop_size
        avail_num = (1:pop_size);
        avail_num(ii) = [];
        % randomly generate three different integers
        r = avail_num(randperm(length(avail_num),2));
        pop_mutation(ii,:) = xmin + F*(pop_vari(r(1),:)-pop_vari(r(2),:));
        % check the bound constraints, randomly re-initialization
        if any(pop_mutation(ii,:)<lower_bound) || any(pop_mutation(ii,:)>upper_bound)
            pop_mutation(ii,:) = lower_bound + rand(1,num_vari).*(upper_bound-lower_bound);
        end
    end
    % crossover
    rand_matrix = rand(pop_size,num_vari);
    temp = randi(num_vari,pop_size,1);
    for ii = 1 : pop_size
        rand_matrix(ii,temp(ii)) = 0;
    end
    mui = rand_matrix < CR;
    pop_trial = pop_mutation.*mui + pop_vari.*(1-mui);
    % select infill samples
    [u,s] = Kriging_Predictor(pop_trial,kriging_model);
    AEI = (pop_obj-u).*gausscdf((pop_obj-u)./s)+s.*gausspdf((pop_obj-u)./s);
    [~,sort_ind] = sort(AEI,'descend');
    select_ind = sort_ind(1:num_q);
    infill_x = pop_trial(select_ind,:);
    infill_y = feval(fun_name,infill_x);
    % replacement
    replace_index =  infill_y <= pop_obj(select_ind,:);
    pop_vari(select_ind(replace_index),:) = infill_x(replace_index,:);
    pop_obj(select_ind(replace_index)) = infill_y(replace_index,:);
    % update the database
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % incremental learning
    kriging_model = Kriging_Incremental_Update(kriging_model,infill_x,infill_y);
    % update the evaluation number of generation number
    generation = generation + 1;
    evaluation = evaluation + size(infill_x,1);
    [fmin,index] = min(sample_y);
    fmin_record(generation,:) = fmin;
    xmin = sample_x(index,:);
    % print the iteration information
    fprintf('KAEA-AEI on %d-D %s, generation: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,generation,evaluation,fmin);
end


