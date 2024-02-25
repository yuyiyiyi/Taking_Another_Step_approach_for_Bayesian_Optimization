clearvars;clc;close all;
% test problem
fun_name = 'Ackley';
num_vari = 100;
max_evaluation = 1000;
num_initial = 2*num_vari;
lower_bound = -32.728*ones(1,num_vari); 
upper_bound = 32.728*ones(1,num_vari);
% hyperparamter: number of nearby points
local_num=2*num_vari;
% number of current generation
generation = 1;
% generate random samples
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation =  size(sample_x,1);
% best objectives in each generation
fmin_record = zeros(max_evaluation-evaluation+1,1);
% update the current minimize value
fmin = min(sample_y);
fmin_record(generation,:) = fmin;
% print the iteration information
fprintf('TAS-BO global min on %s, iteration: %d, evaluation: %d, best: %0.4g\n',fun_name,generation,evaluation,fmin);
% the evoluation of the population
while evaluation < max_evaluation
	% train a global Guassian process (Kriging) model
	kriging_model = Kriging_Train(sample_x,sample_y,lower_bound,upper_bound,1,1E-6,1E2);
	% find the first point using the EI criterion in the global space
	[best_x,max_EI] = Optimizer_GA(@(x)-Infill_EI(x,kriging_model,fmin),num_vari,lower_bound,upper_bound,num_vari,100);
	% train a local Guassian process around the first point
	[~,sort_index] = sort(pdist2(best_x,sample_x));
	local_x = sample_x(sort_index(1:local_num),:);
	local_y = sample_y(sort_index(1:local_num),:);
	kriging_local = Kriging_Train(local_x,local_y,min(local_x),max(local_x),1,1E-6,1E2);
	% find an infill solution using the EI criterion in the local space
	[infill_x,max_EI_local] = Optimizer_GA(@(x)-Infill_EI(x,kriging_local,fmin),num_vari,min(local_x),max(local_x),num_vari,100);
    % evaluate the infill solution
	infill_y = feval(fun_name,infill_x);
    % update the database
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update the evaluation number of generation number and the current minimize value
    generation = generation + 1;
    evaluation = evaluation + size(infill_x,1);
    fmin = min(sample_y);
    fmin_record(generation,:) = fmin;
    % print the iteration information
    fprintf('TAS-BO global min on %s, iteration: %d, evaluation: %d, best: %0.4g\n',fun_name,generation,evaluation,fmin);
end


