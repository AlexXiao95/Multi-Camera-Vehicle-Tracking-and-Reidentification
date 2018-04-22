function [iter,top1,top5] = parse_log(txtName,n_per_line)
% txtName = 'log_88.6.txt.test';
f = fopen(txtName,'r');
fgetl(f);
A = fscanf(f,'%f');

iter = A(1:n_per_line:end);
top1 = A(3:n_per_line:end);
top5 = A(4:n_per_line:end);
len =length(top1);
iter = iter(1:len);
end