% This function generates random matched and unmatched pairs
% im_n: number of samples for each class
% n_match: number of matched pairs
% n_unmatch: number of unmatched pairs 
function [pairs_match, pairs_unmatch] = gen_verif_pairs(im_n, n_match, n_unmatch)

rng(0);%fix rand seed
model_n = size(im_n,1);
im_t = sum(im_n,2);
m_offset = cumsum(im_t);
m_offset = [0;m_offset(1:end-1)];

pairs_try = n_match * 4;
model_s = randi(model_n,[pairs_try,1]);
n_pair_init = randi(10000,pairs_try,2);%init a large rand number

n_pair = mod(n_pair_init,repmat(im_t(model_s),1,2))+1;
% max(n_pa
eq_check = (n_pair(:,1) == n_pair(:,2));
n_pair = sort(n_pair,2);
n_pair = n_pair +repmat(m_offset(model_s),1,2);
uniq_pairs = unique(n_pair(~eq_check,:),'rows');
%note uniq_pairs is sorted
p = randperm(length(uniq_pairs));
pairs_match = uniq_pairs(p(1:n_match),:);
%for pairs in diff models
model_p = randi(model_n,[pairs_try,2]);
eq_check = (model_p(:,1) == model_p(:,2));
% model_p = model_p(~eq_check,:);
model_p = sort(model_p,2);

n_pair_init = randi(10000,pairs_try,2);%init a large rand number
n_pair = mod(n_pair_init,im_t(model_p))+1 + m_offset(model_p);
uniq_pairs = unique(n_pair(~eq_check,:),'rows');
p = randperm(length(uniq_pairs));
pairs_unmatch = uniq_pairs(p(1:n_unmatch),:);
end
